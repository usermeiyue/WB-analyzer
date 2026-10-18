"""Plain scientific labels with Greek escapes and balanced super/subscripts."""
import re
from pathlib import Path
from PIL import ImageFont

FONT_FILES={'Sans serif (DejaVu Sans)':'font.ttf','Serif (DejaVu Serif)':'serif.ttf','Monospace (DejaVu Sans Mono)':'mono.ttf'}
LEGACY_FONT_NAMES={
    '\u65e0\u886c\u7ebf\uff08DejaVu Sans\uff09':'Sans serif (DejaVu Sans)',
    '\u886c\u7ebf\uff08DejaVu Serif\uff09':'Serif (DejaVu Serif)',
    '\u7b49\u5bbd\uff08DejaVu Sans Mono\uff09':'Monospace (DejaVu Sans Mono)',
}

def migrate_font_settings(value):
    if isinstance(value,dict):
        return {k:LEGACY_FONT_NAMES.get(v,v) if isinstance(v,str) and (k=='font_family' or k.endswith('_font_family')) else migrate_font_settings(v) for k,v in value.items()}
    if isinstance(value,list):return [migrate_font_settings(v) for v in value]
    return value

# Use installed proprietary fonts without distributing their files.
for label,candidates in {
    'Arial':['/System/Library/Fonts/Supplemental/Arial.ttf','C:/Windows/Fonts/arial.ttf'],
    'Times New Roman':['/System/Library/Fonts/Supplemental/Times New Roman.ttf','C:/Windows/Fonts/times.ttf'],
    'Helvetica':['/System/Library/Fonts/Helvetica.ttc'],
}.items():
    for candidate in candidates:
        if Path(candidate).is_file():FONT_FILES[label]=candidate;break
GREEK=dict(zip(('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega').split(),'αβγδεζηθικλμνξοπρστυφχψω'))

def greek(text):
    return re.sub(r'\\([A-Za-z]+)',lambda m:GREEK.get(m[1],m[0]),str(text))

FONT_STYLES={'regular':'Regular','bold':'Bold','italic':'Italic','bold_italic':'Bold Italic'}

def font_path(family,style='regular'):
    family=LEGACY_FONT_NAMES.get(family,family)
    filename=FONT_FILES.get(family,'font.ttf');path=Path(filename)
    if not path.is_absolute():path=Path(__file__).resolve().parent.parent/'assets'/filename
    if style=='regular':return path
    if family in list(FONT_FILES)[:3]:return path.with_name(path.stem+'-'+style+path.suffix)
    names={'Arial':{'bold':'Arial Bold.ttf','italic':'Arial Italic.ttf','bold_italic':'Arial Bold Italic.ttf'},'Times New Roman':{'bold':'Times New Roman Bold.ttf','italic':'Times New Roman Italic.ttf','bold_italic':'Times New Roman Bold Italic.ttf'}}
    windows={'Arial':{'bold':'arialbd.ttf','italic':'ariali.ttf','bold_italic':'arialbi.ttf'},'Times New Roman':{'bold':'timesbd.ttf','italic':'timesi.ttf','bold_italic':'timesbi.ttf'}}
    filename=(windows if str(path).startswith('C:') else names).get(family,{}).get(style)
    return path.with_name(filename) if filename else Path('/unavailable-font-style')

def available_styles(family):return [style for style in FONT_STYLES if font_path(family,style).is_file()]

def font_for(size,family,style='regular'):
    path=font_path(family,style)
    if not path.is_file():raise ValueError('This font style is unavailable. Choose Regular or a bundled DejaVu font.')
    return ImageFont.truetype(str(path),max(1,round(size)))

def runs(text,size):
    text=greek(text);out=[];i=0;start=0
    while i<len(text):
        if text[i] in '^_' and i+1<len(text):
            if i>start:out.append((text[start:i],size,0))
            kind=text[i];i+=1
            if text[i]=='{':
                end=text.find('}',i+1)
                if end<0:out.append((text[i-1:],size,0));return out
                value=text[i+1:end];i=end+1
            else:value=text[i];i+=1
            out.append((value,size*.68,-size*.42 if kind=='^' else size*.24));start=i
        else:i+=1
    if start<len(text):out.append((text[start:],size,0))
    return out


def styled_runs(text,size,style='regular',offset=0):
    """Scoped Markdown emphasis, TeX font groups and nested super/subscripts."""
    text=greek(text);out=[];i=0;plain=''
    def flush():
        nonlocal plain
        if plain:out.append((plain,size,offset,style));plain=''
    def group(start):
        depth=1;j=start+1
        while j<len(text):
            if text[j]=='{':depth+=1
            elif text[j]=='}':
                depth-=1
                if depth==0:return text[start+1:j],j+1
            j+=1
        return None,start
    while i<len(text):
        command=re.match(r'\\(textit|italic|textbf|bold|mathrm|textrm|regular)\{',text[i:])
        if command:
            start=i+len(command[0])-1;value,end=group(start)
            if value is not None:
                flush();name=command[1]
                target='regular' if name in ('mathrm','textrm','regular') else ('italic' if name in ('textit','italic') else 'bold')
                out.extend(styled_runs(value,size,target,offset));i=end;continue
        if text[i]=='*':
            count=3 if text.startswith('***',i) else (2 if text.startswith('**',i) else 1)
            token='*'*count;end=text.find(token,i+count)
            if end>i+count:
                flush();bold=style in ('bold','bold_italic') or count>=2;italic=style in ('italic','bold_italic') or count in (1,3)
                target='bold_italic' if bold and italic else ('bold' if bold else 'italic')
                out.extend(styled_runs(text[i+count:end],size,target,offset));i=end+count;continue
        if text[i] in '^_' and i+1<len(text):
            kind=text[i];start=i+1
            value,end=group(start) if text[start]=='{' else (text[start],start+1)
            if value is not None:
                flush();out.extend(styled_runs(value,size*.68,style,offset+(-size*.42 if kind=='^' else size*.24)));i=end;continue
        plain+=text[i];i+=1
    flush();return out


def text_width(text,font):
    family=getattr(font,'family_key',next(iter(FONT_FILES)))
    return sum(font_for(s,family,style).getlength(t) for t,s,_,style in styled_runs(text,font.size,getattr(font,'style_key','regular')))


def scientific_font(size,family,style="regular"):
    f=font_for(size,family,style);f.family_key=family;f.style_key=style;return f


def draw_label(draw,xy,text,font,anchor='lm',fill='black'):
    family=getattr(font,'family_key',next(iter(FONT_FILES)))
    parts=styled_runs(text,font.size,getattr(font,'style_key','regular'))
    width=text_width(text,font);x,y=xy
    if anchor[0]=='m':x-=width/2
    elif anchor[0]=='r':x-=width
    top=min([-font.size]+[offset-s for _,s,offset,_ in parts]);bottom=max([0]+[offset for _,_,offset,_ in parts])
    if anchor[1]=='m':baseline=y-(top+bottom)/2
    elif anchor[1]=='t':baseline=y-top
    else:baseline=y-bottom
    for value,size,offset,style in parts:
        f=font_for(size,family,style);draw.text((x,baseline+offset),value,font=f,fill=fill,anchor='ls');x+=f.getlength(value)
