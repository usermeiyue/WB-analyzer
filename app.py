"""Run: python -m streamlit run app.py"""
from __future__ import annotations
import copy
import io
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from streamlit.components.v1 import declare_component

from wb.core import (Audit, VERSION, canonical, default_protein, display_image, export_excel,
                     export_pdf, export_tiff, figure, image_data_url, load_sources, project_bytes,
                     quant_roi_signature, restore_project, safe_csv, saturation, sha, polarity_label)

from wb.imaging import detect_band_rows, extract_strip, fit_aspect, validate_crop, fixed_crop
from wb.layout import publication_figure
from wb.annotations import FONT_FILES,FONT_STYLES,available_styles
from wb.quant_ui import render_quantification
from wb.source_selection import preferred_source_ids

st.set_page_config(page_title="WB Image Processor & Analyzer",page_icon="🔬",layout="wide")
roi_component=declare_component("wb_roi_editor",path=str(Path(__file__).parent / "roi_component"))
st.markdown("""<style>.block-container{padding-top:2rem}div[data-testid=stMetric]{background:#f0f6fa;padding:14px;border-radius:8px}h1{letter-spacing:-.04em}div[data-testid=stSidebar]{background:#f1f5f9}</style>""",unsafe_allow_html=True)


def initial_samples(n):
    return [{"sample":f"L{i+1}","condition":"Control" if i<2 else "Treatment",
             "bio_rep":str(i+1 if i<2 else i-1),"tech_rep":"1"} for i in range(n)]


def initialize():
    ss=st.session_state
    if "sources" not in ss:
        ss.sources={};ss.proteins=[];ss.samples=initial_samples(6);ss.audit=Audit()
        ss.settings={"reference_id":None,"control_condition":None,"exclude_saturated":True,
                     "dpi":300,"lane_width_mm":7.0,"font_pt":8.0,"gap_mm":2.0,"label_margin_mm":22.0}
        ss.audit.add("session_started",{"application_version":VERSION,"analysis":"unadjusted source pixels"})
        ss.is_demo=False;ss.project_revision=0;ss.loaded=set();ss.last_snapshot=None;ss.exported=None;ss.roi_revision=0;ss.last_roi_nonce=None


def invalidate():
    st.session_state.exported=None


def snapshot():
    ss=st.session_state
    return {"sources":[s.metadata() for s in ss.sources.values()],"proteins":[asdict(p) for p in ss.proteins],
            "samples":ss.samples,"settings":ss.settings}


def checkpoint():
    ss=st.session_state;state=snapshot();fingerprint=sha(canonical(state))
    if fingerprint!=ss.last_snapshot:
        ss.audit.add("parameters_changed",{"snapshot":state,"snapshot_sha256":fingerprint})
        ss.last_snapshot=fingerprint;invalidate()


def demo():
    ss=st.session_state
    # Deterministic synthetic native 16-bit data. Not experimental evidence.
    y,x=np.mgrid[0:260,0:700];rng=np.random.default_rng(14)
    for name,strength in [("Target A",[6000,6200,9000,9500,12000,12500]),("Target B",[8000,8200,6000,6200,4000,4200]),("GAPDH",[7000]*6)]:
        a=2000+rng.normal(0,80,x.shape)
        for lane,amplitude in enumerate(strength):
            a+=amplitude*np.exp(-((x-(100+lane*100))/28)**4-((y-95)/9)**2)
        buf=io.BytesIO();tifffile.imwrite(buf,np.clip(a,0,65535).astype(np.uint16))
        s=load_sources(buf.getvalue(),name.replace(' ','_')+'.tif')[0]
        s.display_white=18000;ss.sources[s.id]=s
        p=default_protein(s,6,name);p.membrane_id='DEMO-M1';p.polarity='light';p.molecular_weight='37 kDa' if name=='GAPDH' else '50 kDa'
        p.crop=[50,65,600,60];p.signal=[[70+i*100,78,60,34] for i in range(6)];p.background=[[70+i*100,145,60,20] for i in range(6)]
        p.quant_roi_signature=quant_roi_signature(p)
        ss.proteins.append(p)
    ss.is_demo=True;ss.samples=initial_samples(6);ss.settings['reference_id']=ss.proteins[-1].id;ss.settings['control_condition']='Control'
    ss.audit.add('synthetic_demo_loaded',{'seed':14,'not_real_experiment':True});ss.roi_revision+=1


initialize();ss=st.session_state
for setting,default in {'strip_height_mm':4.0,'block_lanes':0,'block_gap_mm':1.0,'header_unit':'min','lock_crop_aspect':False,'panel_style':'publication','crop_template':None,'crop_template_owner':None,'subtract_background':True,'font_family':next(iter(FONT_FILES)),'marker_font_pt':7.,'marker_layout':{},'figure_texts':[]}.items():
    ss.settings.setdefault(setting,default)
if ss.settings.get('quantification_version')!=2:
    ss.settings['subtract_background']=True;ss.settings['quantification_version']=2
for role,size in [('lane',ss.settings['font_pt']*.85),('protein',ss.settings['font_pt']),('marker',ss.settings['marker_font_pt'])]:
    ss.settings.setdefault(role+'_font_pt',size)
    ss.settings.setdefault(role+'_font_family',ss.settings['font_family'])
    ss.settings.setdefault(role+'_font_style','regular')
ss.settings.setdefault('kda_font_pt',ss.settings['lane_font_pt'])
ss.settings.setdefault('kda_font_family',ss.settings['lane_font_family'])
ss.settings.setdefault('kda_font_style','regular')
ss.settings.setdefault('kda_layout',{'x_offset_mm':0.,'y_offset_mm':0.})
st.title("WB Image Processor & Analyzer")
st.caption("标注支持直接输入 α、β、γ；上标示例 Pfkl^{S775A/S775A}，下标示例 IL_{1}。")
st.caption("原图定量 · 多蛋白组合图 · 同泳道内参 · 原始文件与参数追溯")
with st.sidebar:
    st.subheader("项目与原图")
    uploads=st.file_uploader("添加 TIFF / PNG / JPG",type=['tif','tiff','png','jpg','jpeg'],accept_multiple_files=True,key='uploads_'+str(ss.project_revision))
    if uploads:
        if ss.get('is_demo',False):
            ss.sources={};ss.proteins=[];ss.loaded=set();ss.is_demo=False;ss.settings['crop_template']=None
            ss.settings['reference_id']=None;ss.settings['control_condition']=None
            ss.project_revision+=1;ss.roi_revision+=1
            ss.audit.add('synthetic_demo_cleared_for_real_upload',{})
        for f in uploads:
            digest=sha(f.getvalue())
            if digest not in ss.loaded:
                try:
                    decoded=load_sources(f.getvalue(),f.name)
                    for src in decoded:
                        if src.id not in ss.sources:
                            ss.sources[src.id]=src
                    ss.loaded.add(digest);ss.audit.add('original_uploaded',{'file':f.name,'sha256':digest,'pages':len(decoded)})
                    ss.roi_revision+=1;invalidate()
                except Exception as error:
                    st.error(f"{f.name}: {error}")
    if st.button("查看示例（非实验数据）",disabled=bool(ss.sources)):
        demo();st.rerun()
    project=st.file_uploader("恢复项目 ZIP",type=['zip'],key='restore_upload')
    if project and st.button("恢复此项目"):
        try:
            ss.sources,ss.proteins,ss.samples,ss.audit,ss.settings=restore_project(project.getvalue())
            # Reset prior widgets before the downstream editor widgets are instantiated.
            retained={'sources','proteins','samples','audit','settings','loaded','last_snapshot','exported','roi_revision','last_roi_nonce','project_revision','restore_upload','is_demo'}
            for key in list(ss):
                if key not in retained and not key.startswith('uploads_'):
                    del ss[key]
            ss.is_demo=any(e['action'] in ['synthetic_demo_loaded','synthetic_demo'] for e in ss.audit.events) and all(p.membrane_id=='DEMO-M1' for p in ss.proteins)
            ss.project_revision+=1
            ss.loaded={sha(s.original) for s in ss.sources.values()};ss.last_snapshot=None;ss.roi_revision+=1;invalidate();st.rerun()
        except Exception as error:
            st.error(str(error))
    st.divider()
    st.caption("本地运行时，文件由本机 Python 进程处理。部署到远程服务器后，上传文件会传给该服务器。当前会话在刷新/关闭后可能丢失，请下载项目。")
    st.caption("审计记录可追溯本工具内的处理，不是防篡改签名或实验真实性认证。")

if not ss.sources:
    st.info("先上传原始图像，或加载示例。每张图可添加多个蛋白，最终输出统一的组合图。")
    st.stop()

if ss.get('is_demo',False):
    st.warning('当前是合成示例：Target A、Target B、GAPDH 都是演示数据。上传你的图片会自动清除示例；也可以点击下面的按钮开始空白实验。')
    if st.button('清空示例，开始自己的实验'):
        ss.sources={};ss.proteins=[];ss.loaded=set();ss.is_demo=False;ss.settings['crop_template']=None
        ss.settings['reference_id']=None;ss.settings['control_condition']=None
        ss.project_revision+=1;ss.roi_revision+=1;invalidate()
        ss.audit.add('synthetic_demo_cleared',{});st.rerun()

preferred_sources=preferred_source_ids(ss.sources)
st.caption(f"① 上传原图 → ② 填写泳道样品 → ③ 添加你要分析的蛋白 → ④ 框选 → ⑤ 归一化与导出 · 已上传 {len(preferred_sources)} 个文件，已添加 {len(ss.proteins)} 个蛋白")
tabs=st.tabs(["1 · 样品","2 · 添加 / 管理蛋白","3 · 图像与选框","4 · 定量与归一化","5 · 组合图与导出","6 · 保存项目"])
with tabs[0]:
    st.subheader("样品与重复设计")
    n=st.number_input("泳道数",min_value=1,max_value=40,value=len(ss.samples))
    if n!=len(ss.samples):
        ss.samples=(ss.samples+initial_samples(n)[len(ss.samples):])[:n]
        for p in ss.proteins:
            fresh=default_protein(ss.sources[p.source_id],n)
            p.signal=fresh.signal;p.background=fresh.background
        ss.audit.add('lane_count_changed',{'lane_count':n,'roi_reset':True});ss.roi_revision+=1;invalidate()
    st.caption('每个泳道只填一次标注。例：标注 un、15、30、60、120；组名都填 LPS。不需要分组标题时，组名留空。')
    for sample in ss.samples:
        if sample.get('figure_label'):sample['sample']=sample.pop('figure_label')
        else:sample.pop('figure_label',None)
        sample.setdefault('figure_group','')
    table=pd.DataFrame([{k:sample.get(k,'') for k in ['sample','figure_group']} for sample in ss.samples])
    edited=st.data_editor(table,hide_index=True,num_rows='fixed',key=f'sample_table_{ss.project_revision}_{n}',column_config={
        'sample':st.column_config.TextColumn('泳道标注'),'figure_group':st.column_config.TextColumn('组名（可选）')})
    for original,changed in zip(ss.samples,edited.fillna('').astype(str).to_dict('records')):original.update(changed)
    with st.expander('定量统计设置（需要均值、SD 时填写）',expanded=False):
        stats=pd.DataFrame([{k:sample.get(k,'') for k in ['condition','bio_rep','tech_rep']} for sample in ss.samples])
        changed=st.data_editor(stats,hide_index=True,key=f'replicates_{ss.project_revision}_{n}',column_config={
            'condition':st.column_config.TextColumn('统计条件'),'bio_rep':st.column_config.TextColumn('生物学重复'),
            'tech_rep':st.column_config.TextColumn('技术复孔')})
        for original,row in zip(ss.samples,changed.fillna('').astype(str).to_dict('records')):original.update(row)

with tabs[1]:
    st.subheader('添加你要分析的蛋白')
    st.write('一个蛋白 = 最终组合图中的一条裁图。只有点击“添加此蛋白”才会创建，不会根据文件名自动生成。')
    source_labels={k:f'{v.name} / 第 {v.page+1} 页 · {v.gray.dtype.itemsize*8}-bit'+(' · 调色板显示页' if 'palette' in v.conversion else '') for k,v in ss.sources.items()}
    show_all_pages=False
    if len(ss.sources)>len(preferred_sources):
        st.caption('每个文件默认显示一页，优先使用原生 16-bit 灰度页；其他页与原始文件仍保留。')
        with st.expander('切换其他图像页（可选）'):
            show_all_pages=st.checkbox('显示全部图像页',key='show_all_source_pages')
    source_options=list(ss.sources) if show_all_pages else preferred_sources
    # Advanced page switching still starts with the preferred native page.
    if ss.get('new_protein_source') not in source_options:
        ss['new_protein_source']=preferred_sources[0]
    with st.form('add_protein_'+str(ss.project_revision),clear_on_submit=True):
        left,right=st.columns(2)
        new_name=left.text_input('蛋白名称',placeholder='输入你实际检测的蛋白名称',key='new_protein_name')
        src_id=right.selectbox('对应原图',source_options,format_func=lambda k:source_labels[k],key='new_protein_source')
        new_mw=left.text_input('分子量（可选）',placeholder='例如 50 kDa',key='new_protein_mw')
        new_membrane=right.text_input('膜编号（归一化时需要）',placeholder='同一张膜的蛋白填写相同编号',key='new_protein_membrane')
        add=st.form_submit_button('添加此蛋白',type='primary')
    if add:
        if not new_name.strip():st.error('请先输入蛋白名称，再点击添加。')
        else:
            new=default_protein(ss.sources[src_id],len(ss.samples),new_name.strip())
            new.molecular_weight=new_mw.strip();new.membrane_id=new_membrane.strip()
            ss.proteins.append(new);ss.audit.add('protein_added',{'protein_id':new.id,'name':new.name,'source_id':new.source_id})
            ss.roi_revision+=1;invalidate();st.rerun()
    st.divider()
    st.subheader(f'已添加的蛋白（{len(ss.proteins)}）')
    st.caption('点“编辑”修改名称、分子量、膜编号或原图；点“删除”移除该蛋白。可以删除全部蛋白，原图和样品信息保留。')
    if not ss.proteins:
        st.info('还没有蛋白。请在上面输入名称、选择原图，然后点击“添加此蛋白”。')
    for i,item in enumerate(list(ss.proteins)):
        with st.container(border=True):
            title,edit_col,delete_col=st.columns([6,1,1])
            title.markdown(f'**{i+1}. {item.name}**')
            title.caption(f"原图：{source_labels[item.source_id]} · 分子量：{item.molecular_weight or '未填'} · 膜编号：{item.membrane_id or '未填'}")
            editing_key='editing_'+item.id
            if edit_col.button('编辑',key='edit_'+item.id):
                ss[editing_key]=not ss.get(editing_key,False)
            if delete_col.button('删除',key='delete_'+item.id):
                ss.proteins=[existing for existing in ss.proteins if existing.id!=item.id]
                if ss.settings['reference_id']==item.id:ss.settings['reference_id']=None
                ss.audit.add('protein_deleted',{'protein_id':item.id,'name':item.name})
                ss.roi_revision+=1;invalidate();st.rerun()
            if ss.get(editing_key,False):
                with st.form('edit_form_'+item.id):
                    c1,c2=st.columns(2)
                    name=c1.text_input('蛋白名称',value=item.name,key='edit_name_'+item.id)
                    mw=c2.text_input('分子量（可选）',value=item.molecular_weight,key='edit_mw_'+item.id)
                    membrane=c1.text_input('膜编号',value=item.membrane_id,key='edit_membrane_'+item.id)
                    edit_options=list(source_options)
                    if item.source_id not in edit_options:edit_options.append(item.source_id)
                    source_id=c2.selectbox('对应原图',edit_options,index=edit_options.index(item.source_id),format_func=lambda k:source_labels[k],key='edit_source_'+item.id)
                    save=st.form_submit_button('保存修改',type='primary')
                if save:
                    if not name.strip():st.error('蛋白名称不能为空。')
                    else:
                        if source_id!=item.source_id:
                            fresh=default_protein(ss.sources[source_id],len(ss.samples),name.strip())
                            item.crop=fresh.crop;item.signal=fresh.signal;item.background=fresh.background;item.crop_angle=0.;item.crop_selected=False;item.segments=[]
                            item.markers=[];item.manual_lines=[];item.segments=[]
                        item.name=name.strip();item.molecular_weight=mw.strip();item.membrane_id=membrane.strip();item.source_id=source_id
                        ss[editing_key]=False;ss.roi_revision+=1;invalidate()
                        ss.audit.add('protein_edited',{'protein_id':item.id,'name':item.name,'source_id':item.source_id});st.rerun()
            a,b,_=st.columns([1,1,6])
            if a.button('上移',key='up_'+item.id,disabled=i==0):
                ss.proteins[i-1],ss.proteins[i]=ss.proteins[i],ss.proteins[i-1];ss.roi_revision+=1;invalidate();st.rerun()
            if b.button('下移',key='down_'+item.id,disabled=i==len(ss.proteins)-1):
                ss.proteins[i+1],ss.proteins[i]=ss.proteins[i],ss.proteins[i+1];ss.roi_revision+=1;invalidate();st.rerun()
    st.caption('膜编号由你核实填写，工具不会猜测目标蛋白或指定 GAPDH 为内参。内参在“定量与归一化”中由你选择。')

if not ss.proteins:
    with tabs[2]:st.info('请先到“2 · 添加 / 管理蛋白”添加一个蛋白，再进行框选。')
    with tabs[3]:st.info('添加实际检测的蛋白后，才能选择内参和定量。')
    with tabs[4]:st.info('添加蛋白后，这里会显示多蛋白组合图。')
    with tabs[5]:
        checkpoint()
        st.download_button('保存当前原图与样品',project_bytes(ss.sources,[],ss.samples,ss.audit,ss.settings),file_name='wb-project.zip',mime='application/zip')
    st.stop()
protein_names=[p.name for p in ss.proteins]
with tabs[2]:
    p_index=st.selectbox("编辑蛋白",range(len(ss.proteins)),format_func=lambda i:protein_names[i],key='active_protein_'+str(ss.project_revision))
    p=ss.proteins[p_index];src=ss.sources[p.source_id];h,w=src.gray.shape
    st.caption(f"原图 {src.name} · 第 {src.page+1} 页 · {w} × {h} · {src.gray.dtype} · {src.conversion}")
    if src.name.lower().endswith(('.jpg','.jpeg')):
        st.warning("JPEG 是有损格式。该文件可处理，但不能恢复原始采集强度；优先使用原始无损 TIFF。")
    operation='crop'
    st.caption('条带和背景定量框统一在“定量与归一化”页面设置。')
    src.display_invert=st.checkbox("反色（黑白翻转）",value=src.display_invert,key='invert_'+src.id)
    st.caption("反色统一应用于该原图的预览和导出；原始像素与灰度定量保持不变。")
    draft_key='crop_draft_'+p.id+'_'+src.id
    if draft_key not in ss:
        ss[draft_key]={'rect':list(p.crop),'angle':p.crop_angle} if p.crop_selected else None
    template=ss.settings['crop_template']
    if template and not any(item.id==ss.settings['crop_template_owner'] for item in ss.proteins):ss.settings['crop_template_owner']=ss.proteins[0].id
    is_owner=p.id==ss.settings['crop_template_owner']
    size_locked=bool(template and not is_owner)
    if size_locked:
        try:
            if ss[draft_key] is None:ss[draft_key]={'rect':fixed_crop(p.crop,template,src.gray.shape),'angle':0.}
            else:ss[draft_key]['rect']=fixed_crop(ss[draft_key]['rect'],template,src.gray.shape,ss[draft_key]['angle'])
        except ValueError as error:st.error(str(error))
    draft=ss[draft_key]
    if operation=='crop':
        if size_locked:st.info(f'固定裁剪尺寸：{template[0]} × {template[1]} 像素。后续框只移动和旋转。')
        elif template:st.info('这是尺寸模板：仍可修改宽、高。确认新尺寸后，其他蛋白的裁剪框同步更新。')
        else:st.info('先调整第一张裁剪框的宽、高。确认后，所有后续框使用这个固定尺寸。')
        padding=st.number_input('检测框四周保留背景 / 原图像素',0,30,3,key='padding_'+p.id)
        if st.button('在原图上检测条带',type='primary',key='detect_'+p.id):
            with st.spinner('正在检测条带位置…'):
                ss['band_candidates_'+src.id]=detect_band_rows(src,padding=int(padding))
            ss.pop('candidate_selection_'+p.id,None)
            ss.audit.add('band_rows_suggested',{'source_id':src.id,'candidates':ss['band_candidates_'+src.id]})
        suggestions=ss.get('band_candidates_'+src.id)
        if suggestions is not None:
            if not suggestions:st.info('未找到清晰的候选条带，请在原图上拖选。')
            else:
                choice=st.selectbox('原图上显示哪一行的裁剪框',range(len(suggestions)),format_func=lambda i:f"第 {i+1} 行 · y={suggestions[i]['source_y']}",key='candidate_'+p.id)
                candidate=suggestions[choice]
                token=(src.id,choice,tuple(candidate['rect']),candidate.get('padding',3))
                if ss.get('candidate_selection_'+p.id)!=token:
                    ss['candidate_selection_'+p.id]=token
                    rect=list(candidate.get('axis_rect',candidate['rect']))
                    if size_locked:rect=fixed_crop(rect,template,src.gray.shape)
                    draft={'rect':rect,'angle':0.,'polarity':candidate['polarity']}
                    ss[draft_key]=draft;ss.roi_revision+=1
                st.caption(f"检测框已显示在下方原图。估计倾斜 {candidate['angle']:g}°；当前不自动旋转、不应用裁剪。")
                if st.button('按估计倾斜角调整框（可选）',key='suggested_rotation_'+p.id):
                    rect=fixed_crop(candidate['rect'],template,src.gray.shape,candidate['angle']) if size_locked else list(candidate['rect'])
                    draft={'rect':rect,'angle':candidate['angle'],'polarity':candidate['polarity']}
                    ss[draft_key]=draft;ss.roi_revision+=1;st.rerun()
        if draft:
            angle=st.slider('旋转待确认裁剪框 / °',-45.,45.,float(draft['angle']),.1,key=f'angle_{p.id}_{ss.roi_revision}')
            if angle!=draft['angle']:
                try:validate_crop(draft['rect'],angle,src.gray.shape);draft['angle']=angle
                except ValueError as error:st.error(str(error))
        st.caption('只改变原图上的待确认框。调整完后，点击原图下方的“确认并使用当前裁剪”。最终各蛋白仍输出等宽等高。')
    with st.expander("饱和检测与原始信号设置",expanded=False):
        ca,cb,cc=st.columns(3)
        p.polarity=ca.selectbox("信号方向",['dark','light'],index=['dark','light'].index(p.polarity),format_func=lambda x:polarity_label(src,x),key=f'polarity_encoding_{p.id}_{ss.roi_revision}_{src.native_inverted}_{src.display_invert}')
        src.floor=cb.number_input("原始传感器下限",value=float(src.floor),key=f'floor_{src.id}')
        src.ceiling=cc.number_input("原始传感器上限",value=float(src.ceiling),key=f'ceiling_{src.id}')
        if src.native_inverted:st.caption("已识别 TIFF 反向灰度编码：选项按当前画面的黑白显示解释，程序自动映射原始像素方向。")
        st.caption("按当前画面选择条带颜色；反色会同步更新选项名称，原始像素与已有定量方向保持不变。16-bit 默认上限 65535；12-bit 采集请填 4095。")
        try:
            mask=saturation(src,p.polarity)
            st.caption(f"疑似饱和：{int(mask.sum()):,} 像素（{mask.mean()*100:.4f}%）。阈值检测不能保证信号处于线性范围。")
        except ValueError as error:st.error(str(error))
        overlay=st.checkbox("红色显示饱和区域（仅屏幕）",value=False,key='overlay_'+p.id)
    with st.expander("整张原图统一展示调整",expanded=False):
        c1,c2=st.columns(2)
        src.display_black=c1.number_input("展示黑点",value=float(src.display_black),key=f'black_{src.id}')
        src.display_white=c2.number_input("展示白点",value=float(src.display_white),key=f'white_{src.id}')
        src.brightness=c1.slider("整体亮度",-.5,.5,float(src.brightness),.01,key=f'brightness_{src.id}')
        src.contrast=c2.slider("整体对比度",.1,3.0,float(src.contrast),.05,key=f'contrast_{src.id}')
        st.caption("线性调整均匀作用于整张原图，所有来自该图的蛋白共享参数。定量不读取这张展示图。")
    try:
        mask=saturation(src,p.polarity)
        mode='crop';lane=0;rectangles=[]
        st.caption("框内拖动可移动，顶部圆点可旋转。尺寸已固定时，框外点击可重新定位；第一张未确认前可调整边界。")
        event=roi_component(image=image_data_url(display_image(src,mask if overlay else None)),raw_width=w,raw_height=h,
                            rectangles=rectangles,mode=mode,lane=lane,revision=ss.roi_revision,
                            crop=draft['rect'] if operation=='crop' and draft else None,angle=draft['angle'] if draft else 0.,
                            aspect=0,fixed_size=template if operation=='crop' and size_locked else None,
                            hint=('固定尺寸：框内移动，顶部圆点旋转，框外点击定位。' if size_locked else '框内移动；顶部圆点旋转；四角或边缘调整大小。') if operation=='crop' else '拖选当前泳道的条带或背景。分隔线模式取框左边界。',key='roi_'+str(ss.project_revision)+'_'+p.id+'_'+operation,default=None)
        if event and event.get('mode') in ['signal','background','crop','separator'] and isinstance(event.get('lane'),int) and 0<=event['lane']<len(ss.samples) and event.get('revision')==ss.roi_revision and event.get('nonce')!=ss.last_roi_nonce:
            ss.last_roi_nonce=event['nonce'];rect=event['rect']
            if event['mode']=='crop':
                angle=float(event.get('angle',0))
                if size_locked:rect=fixed_crop(rect,template,src.gray.shape,angle)
                validate_crop(rect,angle,src.gray.shape)
                ss[draft_key]={'rect':rect,'angle':angle,**({'polarity':draft['polarity']} if draft and 'polarity' in draft else {})}
            elif event['mode']=='separator':p.manual_lines.append(rect[0])
            else:getattr(p,event['mode'])[int(event['lane'])]=rect
            ss.audit.add('crop_draft_changed' if event['mode']=='crop' else 'interactive_roi_changed',{'protein_id':p.id,'mode':event['mode'],'lane':event['lane'],'rect':rect,'angle':float(event.get('angle',0))})
            ss.roi_revision+=1
            if event['mode']!='crop':invalidate()
            st.rerun()
    except ValueError as error:
        st.error(str(error))
    if operation=='crop' and draft:
        st.caption(f"待确认框：{draft['rect'][2]} × {draft['rect'][3]} 原图像素 · {draft['angle']:g}°")
        if st.button('确认并使用当前裁剪',type='primary',key='confirm_crop_'+p.id):
            try:
                validate_crop(draft['rect'],draft['angle'],src.gray.shape)
                if template is None or is_owner:
                    new_size=list(draft['rect'][2:])
                    # Validate every new rectangle first; an incompatible image cannot partially update the project.
                    planned={item.id:fixed_crop(item.crop,new_size,ss.sources[item.source_id].gray.shape,item.crop_angle) for item in ss.proteins if item.id!=p.id}
                    draft_plans={}
                    for item in ss.proteins:
                        key='crop_draft_'+item.id+'_'+item.source_id
                        if item.id!=p.id and ss.get(key):draft_plans[key]=fixed_crop(ss[key]['rect'],new_size,ss.sources[item.source_id].gray.shape,ss[key]['angle'])
                    ss.settings['crop_template']=new_size;ss.settings['crop_template_owner']=p.id
                    for item in ss.proteins:
                        if item.id==p.id:continue
                        item.crop=planned[item.id];item.segments=[]
                        other_key='crop_draft_'+item.id+'_'+item.source_id
                        if other_key in draft_plans:ss[other_key]['rect']=draft_plans[other_key]
                    ss.audit.add('crop_template_updated',{'owner':p.id,'size':new_size,'other_crops':planned})
                p.crop=list(draft['rect']);p.crop_angle=float(draft['angle']);p.crop_selected=True
                p.segments=[]
                if 'polarity' in draft:p.polarity=draft['polarity']
                ss.audit.add('crop_confirmed',{'protein_id':p.id,'crop':p.crop,'angle':p.crop_angle})
                ss.roi_revision+=1;invalidate();st.rerun()
            except ValueError as error:st.error(str(error))
    if p.crop_selected:
        with st.expander('已确认裁剪结果（用于最终组合图）',expanded=False):
            try:st.image(extract_strip(src,p))
            except ValueError as error:st.error(str(error))
    with st.expander("Marker、非连续泳道和分隔线"):
        st.caption("裁图时保留必要背景和原始 Marker；这里的 kDa 文字不生成或替代真实 Marker。非连续泳道裁成区段后，程序强制添加拼接分隔线。")
        st.caption('Marker 位置可在组合图页面调整。多条 Marker 可在这里输入原图 y 坐标，组合图会按裁剪位置映射。')
        segments=st.data_editor(pd.DataFrame(p.segments,columns=['x','width']),num_rows='dynamic',hide_index=True,key='segments_'+p.id)
        p.segments=[[int(row.x),int(row.width)] for row in segments.itertuples() if pd.notna(row.x) and pd.notna(row.width)]
        st.caption("留空 = 连续 crop；区段必须按原图从左到右排列，且位于 crop 内。")
        lines=st.text_input("手动分隔线原图 x 坐标，用逗号分隔",value=','.join(map(str,p.manual_lines)),key='lines_'+p.id+'_'+str(ss.roi_revision))
        try:p.manual_lines=[int(x.strip()) for x in lines.split(',') if x.strip()]
        except ValueError:st.error('分隔线坐标必须是整数。')


for field in ['dpi','lane_font_pt','protein_font_pt','marker_font_pt','lane_font_family','protein_font_family','marker_font_family','lane_font_style','protein_font_style','marker_font_style']:
    widget_key='layout_'+field+'_'+str(ss.project_revision)
    if widget_key in ss:ss.settings[field]=ss[widget_key]
ss.settings['font_family']=ss.settings['protein_font_family'];ss.settings['font_pt']=ss.settings['protein_font_pt']
chart_files={}
with tabs[3]:
    raw,bio,summary,chart_files=render_quantification(ss,roi_component)

ref_names={item.id:item.name for item in ss.proteins}
with tabs[4]:
    st.subheader("多蛋白组合图")
    controls=st.columns(2)
    ss.settings['dpi']=controls[0].selectbox('DPI',[300,600],index=[300,600].index(ss.settings['dpi']),key='layout_dpi_'+str(ss.project_revision))
    ss.settings['gap_mm']=controls[1].number_input('蛋白条间距 / mm',.2,15.,float(ss.settings['gap_mm']),.2)
    families=list(FONT_FILES)
    text_columns=st.columns(3)
    for col,role,label in zip(text_columns,['lane','protein','marker'],['顶部泳道','右侧蛋白名','Marker']):
        col.markdown('**'+label+'标注**')
        family=role+'_font_family';size=role+'_font_pt'
        ss.settings[family]=col.selectbox(label+'字体',families,index=families.index(ss.settings[family]) if ss.settings[family] in families else 0,key='layout_'+family+'_'+str(ss.project_revision))
        ss.settings[size]=col.number_input(label+'字号 / pt',1.,72.,float(ss.settings[size]),.1,key='layout_'+size+'_'+str(ss.project_revision))
        styles=available_styles(ss.settings[family]);style_key=role+'_font_style';current=ss.settings.get(style_key,'regular')
        ss.settings[style_key]=col.selectbox(label+'字形',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k],key='layout_'+style_key+'_'+str(ss.project_revision))
    ss.settings['font_family']=ss.settings['protein_font_family'];ss.settings['font_pt']=ss.settings['protein_font_pt']
    st.caption(r'局部格式：*Pfkl*^{S775A/S775A} 仅基因名斜体；**文字** 加粗；\mathrm{文字} 局部常规。整体选 Regular 时其余文字保持常规。')
    st.caption('顶部组名和单位跟随泳道字体；三处字号可分别调整。已安装的 Arial、Helvetica、Times New Roman 会自动列入字体选项。')
    with st.expander('kDa 单位：独立字体、字号和位置',expanded=False):
        kcols=st.columns(3)
        ss.settings['kda_font_family']=kcols[0].selectbox('kDa 字体',families,index=families.index(ss.settings['kda_font_family']) if ss.settings['kda_font_family'] in families else 0)
        ss.settings['kda_font_pt']=kcols[1].number_input('kDa 字号 / pt',1.,72.,float(ss.settings['kda_font_pt']),.1)
        styles=available_styles(ss.settings['kda_font_family']);current=ss.settings['kda_font_style']
        ss.settings['kda_font_style']=kcols[2].selectbox('kDa 字形',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k])
        pos=st.columns(2);kd=ss.settings['kda_layout']
        kd['x_offset_mm']=pos[0].number_input('kDa 左右偏移 / mm',-100.,100.,float(kd['x_offset_mm']),.1)
        kd['y_offset_mm']=pos[1].number_input('kDa 上下偏移 / mm',-100.,100.,float(kd['y_offset_mm']),.1)
        st.caption('相对于默认位置移动；正值向右/向下，负值向左/向上。不会改变 Marker 数字的位置。')
    ss.settings['header_unit']=st.text_input('顶部单位（可留空）',value=ss.settings['header_unit'])
    with st.expander('Marker 位置',expanded=False):
        st.caption('0% 为裁条顶部，100% 为底部。调整的是文字和刻度，不会移动原图的 Marker 条带。')
        marker_protein=st.selectbox('调整哪条蛋白的 Marker',[item.id for item in ss.proteins],format_func=lambda key:ref_names[key])
        marker_item=next(item for item in ss.proteins if item.id==marker_protein)
        style=ss.settings['marker_layout'].setdefault(marker_protein,{'position_pct':50.,'x_offset_mm':0.})
        style['position_pct']=st.slider('Marker 上下位置 / %',0.,100.,float(style['position_pct']),1.,key='marker_pos_'+marker_protein)
        style['x_offset_mm']=st.slider('Marker 左右偏移 / mm',-5.,.3,float(style['x_offset_mm']),.1,key='marker_x_'+marker_protein)
        if st.button('添加 Marker 标注',key='add_marker_'+marker_protein):
            if not marker_item.markers and marker_item.molecular_weight:
                marker_item.markers=[{'label':marker_item.molecular_weight.replace('kDa','').strip(),'position_pct':style['position_pct'],'x_offset_mm':style['x_offset_mm']}]
            marker_item.markers.append({'label':'','position_pct':50.,'x_offset_mm':style['x_offset_mm']})
        if marker_item.markers:
            st.caption('此蛋白有多条 Marker，请分别调整下面的位置。')
            x,y,w,h=marker_item.crop
            entries=[{'label':m['label'],'position_pct':m.get('position_pct',(m.get('y',y+h/2)-y)/h*100),'x_offset_mm':m.get('x_offset_mm',style['x_offset_mm'])} for m in marker_item.markers]
            edited=st.data_editor(pd.DataFrame(entries),hide_index=True,num_rows='dynamic',key='marker_panel_'+marker_protein,column_config={'position_pct':st.column_config.NumberColumn('上下位置 %',min_value=0,max_value=100),'x_offset_mm':st.column_config.NumberColumn('左右偏移 mm',min_value=-5,max_value=.3),'label':'Marker 文字'})
            marker_item.markers=[{'label':str(row.label),'position_pct':float(row.position_pct),'x_offset_mm':float(row.x_offset_mm)} for row in edited.itertuples() if pd.notna(row.label) and pd.notna(row.position_pct) and pd.notna(row.x_offset_mm)]
    st.markdown('**自由文字标注**')
    if st.button('添加文字',key='add_figure_text'):
        ss.settings['figure_texts'].append({'id':uuid.uuid4().hex,'text':'','x_pct':50.,'y_pct':10.,'font_pt':ss.settings['protein_font_pt'],'font_family':ss.settings['protein_font_family']})
    if ss.settings['figure_texts']:
        st.caption('支持希腊字母和上下标。位置百分比以整张组合图为基准；可添加多条文字。')
    for index,item in enumerate(list(ss.settings['figure_texts'])):
        tid=item['id']
        with st.expander('文字 '+str(index+1)+((' · '+item['text']) if item['text'] else ''),expanded=True):
            item['text']=st.text_input('文字内容',value=item['text'],key='free_text_'+tid)
            position=st.columns(2)
            item['x_pct']=position[0].slider('文字左右位置 / %',0.,100.,float(item['x_pct']),.1,key='free_x_'+tid)
            item['y_pct']=position[1].slider('文字上下位置 / %',0.,100.,float(item['y_pct']),.1,key='free_y_'+tid)
            style=st.columns(2)
            item['font_family']=style[0].selectbox('文字字体',families,index=families.index(item['font_family']) if item['font_family'] in families else 0,key='free_font_'+tid)
            item['font_pt']=style[1].number_input('文字字号 / pt',1.,72.,float(item['font_pt']),.1,key='free_size_'+tid)
            styles=available_styles(item['font_family']);current=item.get('font_style','regular')
            item['font_style']=st.selectbox('文字字形',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k],key='free_style_'+tid)
            if st.button('删除这条文字',key='free_delete_'+tid):
                ss.settings['figure_texts']=[other for other in ss.settings['figure_texts'] if other['id']!=tid]
                invalidate();st.rerun()
    with st.expander('更多排版设置',expanded=False):
        ss.settings['label_margin_mm']=st.number_input('蛋白名留白 / mm',10.,60.,float(ss.settings['label_margin_mm']),1.)
        ss.settings['block_lanes']=st.number_input('每块泳道数（0 = 不分块）',0,len(ss.samples),min(int(ss.settings['block_lanes']),len(ss.samples)))
        ss.settings['block_gap_mm']=st.number_input('图块间距 / mm',.2,5.,float(ss.settings['block_gap_mm']),.2)
    st.caption('裁片尺寸由第一张裁剪决定。组合时只垂直排列，不按泳道数改宽高；600 DPI 对所有裁片使用同一倍率。')
    try:
        options={k:ss.settings[k] for k in ['dpi','font_pt','gap_mm','label_margin_mm','block_lanes','block_gap_mm','header_unit','font_family','marker_font_pt','marker_layout','lane_font_pt','protein_font_pt','lane_font_family','protein_font_family','marker_font_family','figure_texts','lane_font_style','protein_font_style','marker_font_style','kda_font_pt','kda_font_family','kda_font_style','kda_layout']}
        panel,panel_meta=publication_figure(ss.sources,ss.proteins,ss.samples,**options,native_size=True)
        st.image(panel,caption=f"{panel.width} × {panel.height} px · {ss.settings['dpi']} DPI · 每蛋白独立裁图，边框/留白披露组合",width='stretch')
        if any(r['upsampled'] for r in panel_meta['strips']):st.warning('部分原图裁片被放大；DPI 和插值不会增加原始细节。请检查目标期刊要求。')
        st.caption("组合 TIFF 为无损 LZW 压缩的 8-bit RGB 展示图；原始 16-bit 文件原样保存在项目 ZIP 中。PDF 的物理页宽与 TIFF 保持一致。")
        checkpoint()
        if st.button("生成完整导出包",type='primary'):
            artifacts={'figure.tif':export_tiff(panel,ss.settings['dpi']),'figure.pdf':export_pdf(panel,ss.settings['dpi']),
                       'figure_metadata.json':canonical(panel_meta)}
            artifacts.update(chart_files)
            if raw is not None:
                artifacts.update({'measurements.csv':safe_csv(raw),'biological_replicates.csv':safe_csv(bio),
                                  'mean_sd.csv':safe_csv(summary),'analysis.xlsx':export_excel(raw,bio,summary,ss.quant_result['details']),
                                  'all_intensities.csv':safe_csv(ss.quant_result['details'])})
            ss.audit.add('exports_generated',{'files':{name:sha(data) for name,data in artifacts.items()},'figure_metadata':panel_meta,'quantification_included':raw is not None})
            audit_bytes=canonical(ss.audit.document(ss.settings,[s.metadata() for s in ss.sources.values()]))
            project=project_bytes(ss.sources,ss.proteins,ss.samples,ss.audit,ss.settings,artifacts)
            ss.exported={**artifacts,'audit.json':audit_bytes,'wb-project-and-results.zip':project}
        if ss.exported:
            d1,d2,d3,d4=st.columns(4)
            for col,name,mime in [(d1,'wb-project-and-results.zip','application/zip'),(d2,'figure.tif','image/tiff'),(d3,'figure.pdf','application/pdf'),(d4,'analysis.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')]:
                if name in ss.exported:col.download_button(name,ss.exported[name],file_name=name,mime=mime)
            for name in ['measurements.csv','mean_sd.csv','audit.json']:
                if name in ss.exported:st.download_button(name,ss.exported[name],file_name=name)
    except Exception as error:
        st.error('无法生成组合图：'+str(error))
    st.caption("300/600 DPI 是可配置输出规格，并不保证符合所有 SCI 期刊；请核对目标期刊的尺寸、格式、原图提交与拼接披露要求。")

with tabs[5]:
    checkpoint()
    st.subheader('保存可继续编辑的项目')
    st.caption('保存原图、裁剪框、标注和定量设置。下次上传项目 ZIP 即可继续。')
    data=project_bytes(ss.sources,ss.proteins,ss.samples,ss.audit,ss.settings)
    st.download_button('保存可恢复项目（含原图）',data,file_name='wb-project.zip',mime='application/zip')
    with st.expander('处理记录（可选查看）',expanded=False):
        st.caption('这里自动记录原图来源、裁剪坐标、旋转、反色及定量参数，便于回查；不是对实验真实性的认证。')
        if not ss.audit.verify():st.warning('旧处理记录不完整，不能据此判断原图是否有问题。')
        st.json(ss.audit.document(ss.settings,[src.metadata() for src in ss.sources.values()]),expanded=False)
