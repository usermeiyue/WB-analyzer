"""Fixed-size publication panels with grouped headings and right protein labels."""
from __future__ import annotations
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .core import display_image, get_font, rect_slice, MAX_PIXELS
from .imaging import extract_strip
from .annotations import scientific_font, text_width, draw_label


def publication_figure(sources,proteins,samples,*,dpi=300,lane_width_mm=7.,font_pt=8.,gap_mm=1.,label_margin_mm=25.,strip_height_mm=4.,block_lanes=0,block_gap_mm=1.,header_unit='min',native_size=False,font_family='Sans serif (DejaVu Sans)',marker_font_pt=None,marker_layout=None,lane_font_pt=None,protein_font_pt=None,lane_font_family=None,protein_font_family=None,marker_font_family=None,figure_texts=None,lane_font_style="regular",protein_font_style="regular",marker_font_style="regular",kda_font_style="regular",kda_font_pt=None,kda_font_family=None,kda_layout=None):
    if dpi not in (300,600):raise ValueError('DPI must be 300 or 600.')
    if not proteins or not samples:raise ValueError('Add proteins and samples first.')
    if strip_height_mm<=0 or lane_width_mm<=0:raise ValueError('Strip height and lane width must be positive.')
    unit=dpi/25.4;n=len(samples);lane_w=max(1,round(lane_width_mm*unit));strip_h=max(1,round(strip_height_mm*unit))
    content_width=n*lane_w
    if native_size:
        source_size=proteins[0].crop[2:]
        if any(p.crop[2:]!=source_size or p.segments for p in proteins):raise ValueError("Confirm every protein using the first crop's fixed dimensions. Figure scaling is not used to equalize strip sizes.")
        content_width=max(1,round(source_size[0]*dpi/300));strip_h=max(1,round(source_size[1]*dpi/300));lane_w=content_width/n
    block_lanes=int(block_lanes) or n
    blocks=[(i,min(i+block_lanes,n)) for i in range(0,n,block_lanes)]
    block_gap=max(1,round(block_gap_mm*unit));left=round(11*unit);right=round(label_margin_mm*unit)
    font=scientific_font(max(1,round((protein_font_pt or font_pt)*dpi/72)),protein_font_family or font_family,protein_font_style)
    small=scientific_font(max(1,round((lane_font_pt or font_pt*.85)*dpi/72)),lane_font_family or font_family,lane_font_style)
    header_font=scientific_font(max(1,round((lane_font_pt or font_pt)*dpi/72)),lane_font_family or font_family,lane_font_style)
    marker_font=scientific_font(max(1,round((marker_font_pt or font_pt*.85)*dpi/72)),marker_font_family or font_family,marker_font_style);marker_layout=marker_layout or {}
    kda_font=scientific_font(max(1,round((kda_font_pt or lane_font_pt or font_pt)*dpi/72)),kda_font_family or lane_font_family or font_family,kda_font_style)
    for item in proteins:
        entry=marker_layout.get(item.id,{})
        labels=item.markers or ([{'label':re.sub(r'\s*kDa\s*','',item.molecular_weight,flags=re.I)}] if item.molecular_weight else [])
        for marker in labels:left=max(left,round(text_width(str(marker['label']),marker_font)+10-float(marker.get('x_offset_mm',entry.get('x_offset_mm',0)))*unit))
    left=max(left,round(text_width("kDa",kda_font)+8));right=max(right,round(max(text_width(p.name,font) for p in proteins)+12))
    gap=max(1,round(gap_mm*unit));top=round(max(header_font.size,small.size,kda_font.size)*3.4);width=round(left+content_width+block_gap*(len(blocks)-1)+right)
    row_extent=max(strip_h,font.size*1.6,marker_font.size*1.6);height=top+round(len(proteins)*row_extent)+gap*(len(proteins)-1)+round(2*unit)
    if width*height>MAX_PIXELS:raise ValueError('Figure exceeds 40 million pixels. Reduce its dimensions.')
    result=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(result);line=max(1,round(dpi/300));meta=[]
    lane_x=[left+i*lane_w+(i//block_lanes)*block_gap for i in range(n)]
    group_labels=[(s.get('figure_group') or ('' if native_size else s.get('condition',''))) for s in samples]
    for b0,b1 in blocks:
        i=b0
        while i<b1:
            j=i+1
            while j<b1 and group_labels[j]==group_labels[i]:j+=1
            label=str(group_labels[i]);x0=lane_x[i];x1=lane_x[j-1]+lane_w
            if label and not (j-i==1 and label==str(samples[i].get('figure_label','') or samples[i]['sample'])):
                if text_width(label,header_font)>x1-x0-4:raise ValueError('Group heading is too long. Shorten it or reduce font size.')
                draw_label(draw,((x0+x1)/2,header_font.size*.25),label,font=header_font,fill='black',anchor='mt')
                draw.line((x0+4,header_font.size*1.65,x1-4,header_font.size*1.65),fill='black',width=line)
            i=j
    for i,sample in enumerate(samples):
        label=str(sample.get('figure_label','') or sample['sample'])
        if text_width(label,small)>lane_w*.96:raise ValueError('Lane label is too long. Shorten it or reduce font size.')
        draw_label(draw,(lane_x[i]+lane_w/2,top-small.size*.35),label,font=small,fill='black',anchor='mb')
    kd=kda_layout or {};kx=left-3+float(kd.get('x_offset_mm',0))*unit;ky=top-small.size*.35+float(kd.get('y_offset_mm',0))*unit
    if kx-text_width('kDa',kda_font)<0 or kx>width or ky-kda_font.size<0 or ky>height:raise ValueError('kDa label exceeds figure bounds. Adjust its position or font size.')
    draw_label(draw,(kx,ky),'kDa',font=kda_font,fill='black',anchor='rb')
    if header_unit:draw_label(draw,(width-right+4,top-small.size*.35),'['+header_unit+']',font=small,fill='black',anchor='lb')
    for ri,p in enumerate(proteins):
        source=sources[p.source_id];x,y,w,h=p.crop
        if p.segments:
            if p.crop_angle:raise ValueError('Rotated strips do not support noncontiguous segments. Add separate entries.')
            rect_slice(p.crop,source.gray.shape);parts=[];previous=x
            for xx,ww in p.segments:
                rect_slice([xx,y,ww,h],source.gray.shape)
                if xx<previous or xx<x or xx+ww>x+w:raise ValueError('Noncontiguous segments must be inside the crop, ordered left to right without overlap.')
                previous=xx+ww;parts.append(display_image(source).crop((xx,y,xx+ww,y+h)))
            strip=Image.new('RGB',(sum(part.width for part in parts),h),'white');offset=0;seams=[]
            for part in parts:
                if offset:seams.append(offset)
                strip.paste(part,(offset,0));offset+=part.width
        else:strip=extract_strip(source,p);seams=[]
        # Uniform scaling, centered on a fixed canvas. Never squeeze a band vertically.
        scale=(dpi/300) if native_size else min(content_width/strip.width,strip_h/strip.height)
        rw=max(1,round(strip.width*scale));rh=max(1,round(strip.height*scale))
        arr=cv2.resize(np.asarray(strip),(rw,rh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        canvas=Image.new('RGB',(content_width,strip_h),'white');ox=(content_width-rw)//2;oy=(strip_h-rh)//2;canvas.paste(Image.fromarray(arr),(ox,oy))
        row_y=round(top+ri*(row_extent+gap)+(row_extent-strip_h)/2)
        for b0,b1 in blocks:
            bx=round(lane_x[b0]);bw=round(b1*lane_w)-round(b0*lane_w)
            result.paste(canvas.crop((round(b0*lane_w),0,round(b1*lane_w),strip_h)),(bx,row_y))
            draw.rectangle((bx,row_y,bx+bw-1,row_y+strip_h-1),outline='black',width=line)
        for seam in seams:
            px=ox+seam*scale;lane=min(n-1,int(px/lane_w));xx=left+px+(lane//block_lanes)*block_gap
            draw.line((xx,row_y,xx,row_y+strip_h-1),fill='black',width=max(2,line))
        for pos in p.manual_lines:
            if p.crop_angle:raise ValueError('Use block layout for separators in rotated strips; source-image vertical lines cannot be mapped directly.')
            if not p.segments and x<=pos<x+w:
                px=ox+(pos-x)*scale;xx=left+px+int(px/lane_w)//block_lanes*block_gap
                draw.line((xx,row_y,xx,row_y+strip_h-1),fill='black',width=max(2,line))
        if text_width(p.name,font)>right-8:raise ValueError('Protein name exceeds the right margin. Increase label margin or reduce font size.')
        center=row_y+strip_h/2;draw_label(draw,(width-right+6,center),p.name,font=font,fill='black',anchor='lm')
        mw=re.sub(r'\s*kDa\s*','',p.molecular_weight,flags=re.I)
        style=marker_layout.get(p.id,{})
        markers=p.markers or ([{'label':mw,'position_pct':style.get('position_pct',50)}] if mw else [])
        placed=[]
        for marker in markers:
            position=float(marker.get('position_pct',(float(marker.get('y',y+h/2))-y)/h*100))
            if not 0<=position<=100:raise ValueError('Marker position must be within 0–100% of strip height.')
            marker_y=row_y+position/100*(strip_h-1)
            marker_x=left-6+float(marker.get('x_offset_mm',style.get('x_offset_mm',0)))*unit
            label=str(marker['label'])
            if marker_x-text_width(label,marker_font)<0 or marker_x>left-2:raise ValueError('Marker label exceeds the left margin. Adjust horizontal position or font size.')
            draw_label(draw,(marker_x,marker_y),label,font=marker_font,fill='black',anchor='rm')
            draw.line((left-4,marker_y,left,marker_y),fill='black',width=line)
            placed.append({'label':label,'position_pct':position,'output_x':marker_x,'output_y':marker_y})
        meta.append({'protein':p.name,'source_id':p.source_id,'crop':p.crop,'crop_angle':p.crop_angle,'segments':p.segments,
                     'markers':placed,'output_y':row_y,'output_width':content_width,'output_height':strip_h,'scale':scale,'upsampled':scale>1.001,
                     'padding_pixels':[ox,oy],'block_cuts':blocks[1:],'source_display':source.metadata()})
    text_meta=[]
    for item in figure_texts or []:
        label=str(item.get('text',''))
        if not label.strip():continue
        size=float(item.get('font_pt',font_pt));family=item.get('font_family',font_family)
        extra_font=scientific_font(max(1,round(size*dpi/72)),family,item.get("font_style","regular"))
        text_w=text_width(label,extra_font)
        if text_w>width-4:raise ValueError('Free text is too long. Shorten it or reduce its font size.')
        xp=float(item.get('x_pct',50));yp=float(item.get('y_pct',10))
        if not 0<=xp<=100 or not 0<=yp<=100:raise ValueError('Free-text position must be within 0–100%.')
        # Keep the complete label within the canvas even at its edges.
        xx=max(text_w/2+2,min(width-text_w/2-2,width*xp/100))
        yy=max(extra_font.size,min(height-extra_font.size,height*yp/100))
        draw_label(draw,(xx,yy),label,font=extra_font,fill='black',anchor='mm')
        text_meta.append({**item,'output_x':xx,'output_y':yy})
    return result,{'dpi':dpi,'pixels':list(result.size),'physical_mm':[result.width/unit,result.height/unit],
                   'native_size':native_size,'layout':'group headings / left kDa / right protein names','strip_height_mm':strip_height_mm,
                   'lane_width_mm':lane_width_mm,'font_pt':font_pt,'font_family':font_family,'marker_font_pt':marker_font_pt,'marker_layout':marker_layout,'block_lanes':block_lanes,
                   'text_styles':{'lanes':{'font_family':lane_font_family or font_family,'font_pt':lane_font_pt or font_pt*.85},'proteins':{'font_family':protein_font_family or font_family,'font_pt':protein_font_pt or font_pt},'markers':{'font_family':marker_font_family or font_family,'font_pt':marker_font_pt or font_pt*.85}},
                   'kda_annotation':{'font_pt':kda_font_pt or lane_font_pt or font_pt,'font_family':kda_font_family or lane_font_family or font_family,'font_style':kda_font_style,'output_x':kx,'output_y':ky,'layout':kd},
                   'font_styles':{'lanes':lane_font_style,'proteins':protein_font_style,'markers':marker_font_style},
                   'interpolation':'uniform geometric rotation and proportional resizing; raw analysis unchanged',
                   'figure_texts':text_meta,'strips':meta}
