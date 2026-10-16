"""Measured lane ratios or biological replicate means; never infer significance."""
import io
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from .annotations import FONT_FILES,greek,font_path,styled_runs


def chart_label(value):
    value=str(value)
    if not any(token in value for token in ('*','\\textit{','\\italic{','\\textbf{','\\bold{','\\mathrm{','\\textrm{','\\regular{')):
        return re.sub(r'([_^])\{([^{}]+)\}',lambda m:'$'+m[1]+'{'+m[2]+'}$',greek(value))
    pieces=[]
    for text,size,offset,style in styled_runs(value,100):
        escaped=''.join({' ':r'\ ', '_':r'\_', '%':r'\%', '$':r'\$', '{':r'\{', '}':r'\}'}.get(c,c) for c in text)
        command={'regular':'mathrm','italic':'mathit','bold':'mathbf','bold_italic':'mathbfit'}[style]
        part='\\'+command+'{'+escaped+'}'
        if offset:part=('^' if offset<0 else '_')+'{'+part+'}'
        pieces.append(part)
    return '$'+''.join(pieces)+'$'


def normalization_chart(raw,bio,protein_id,reference_name,*,grouped=False,dpi=300,font_family='无衬线（DejaVu Sans）',font_pt=9,intensity_only=False,font_style="regular"):
    rows=(bio if grouped else raw)
    rows=rows[rows.protein_id==protein_id].copy()
    value='bio_value' if grouped else ('analysis_intensity' if intensity_only else 'normalized_ratio')
    rows=rows[np.isfinite(rows[value].astype(float))]
    if rows.empty:raise ValueError('当前蛋白没有可用比值；请先核实条带与内参。')
    prop=FontProperties(fname=str(font_path(font_family,font_style)),size=font_pt)
    fig=Figure(figsize=(max(4,min(12,len(rows)*.55)),3.6),dpi=dpi);ax=fig.subplots()
    if grouped:
        groups=list(rows.condition.unique());means=[];sds=[];points=[]
        for condition in groups:
            vals=rows.loc[rows.condition==condition,value].to_numpy(float)
            means.append(vals.mean());sds.append(vals.std(ddof=1) if len(vals)>1 else 0);points.append(vals)
        ax.bar(range(len(groups)),means,yerr=sds,capsize=4,color='#b5c5d5',edgecolor='black',width=.65)
        for i,vals in enumerate(points):ax.scatter(i+np.linspace(-.12,.12,len(vals)) if len(vals)>1 else [i],vals,color='black',s=22,zorder=3)
        labels=groups;caption='均值 ± SD；每个点为一个生物学重复（技术复孔先取均值）。n=1 时不显示 SD。'
    else:
        vals=rows[value].to_numpy(float);ax.bar(range(len(rows)),vals,color='#879fba',edgecolor='black',width=.65)
        labels=rows['sample'].tolist();caption='每根柱为一个泳道的靶蛋白/对应泳道内参比值；未添加误差棒或显著性检验。'
    if intensity_only and not grouped:caption='每根柱为一个泳道的背景扣除条带强度；当前未选择内参。'
    marked=raw[raw.protein_id==protein_id]
    manual_count=int(marked['zero_override_applied'].fillna(False).sum()) if 'zero_override_applied' in marked else 0
    auto_count=int((marked['zero_assignment_source']=='automatic_below_detection_threshold').sum()) if 'zero_assignment_source' in marked else 0
    count=manual_count+auto_count
    if count:
        caption+='按 0 计算：自动未检出 '+str(auto_count)+' 个泳道，用户手动标记 '+str(manual_count)+' 个泳道。'
        fig.text(.02,.015,'Set to 0: '+str(auto_count)+' auto-undetected lane(s); '+str(manual_count)+' manually marked lane(s).',fontproperties=prop)
    ax.set_xticks(range(len(labels)),[chart_label(x) for x in labels],rotation=45 if len(labels)>6 else 0,ha='right' if len(labels)>6 else 'center',fontproperties=prop)
    ax.set_ylabel(chart_label(rows.protein.iloc[0])+(' intensity' if intensity_only else ' / '+chart_label(reference_name)),fontproperties=prop)
    for tick in ax.get_yticklabels():tick.set_fontproperties(prop)
    ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='black',linewidth=.6)
    fig.tight_layout(rect=(0,.09 if count else 0,1,1));files={}
    for fmt in ['png','pdf']:
        buf=io.BytesIO();fig.savefig(buf,format=fmt,dpi=dpi);files['normalization.'+fmt]=buf.getvalue()
    return files,caption
