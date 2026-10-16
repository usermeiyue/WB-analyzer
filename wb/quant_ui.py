"""Independent choose / edit / calculate / download workflow."""
import copy
import io
import json
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from .core import display_image,image_data_url,rect_slice,safe_csv,export_excel
from .quantification import empty_workspace,propose_workspace,calculate,quant_fingerprint,suggest_background,match_first_lane_size,background_for_roi
from .charts import normalization_chart


def status_label(value):
    names={'invalid_or_nonpositive_reference':'内参无有效正强度，无法归一化','missing_signal_or_background':'缺少条带框或无法估计背景',
           'background_overlaps_signal':'条带框与背景框重叠','saturated_signal_or_background':'条带或背景存在饱和像素',
           'pixels_outside_sensor_range':'像素超出传感器范围','membrane_pairing_mismatch':'靶蛋白与内参膜编号不同'}
    parts=[names.get(v,v) for v in str(value).split(';') if v!='ok']
    return '；'.join(parts) if parts else '计算完成'


def touch(ss,workspace):
    workspace['revision']+=1;ss.exported=None


def remember(ss,p,workspace):
    history=ss.setdefault('quant_undo',[])
    history.append({'protein_id':p.id,'workspace':copy.deepcopy(workspace)})
    del history[:-100]


def band_view(source,workspace,backgrounds=None):
    h,w=source.gray.shape;rects=[r['signal'] for r in workspace['lanes'] if r.get('signal')]
    rects+=list(backgrounds or [])
    if not rects:return [0,0,w,h]
    left=max(0,min(r[0] for r in rects)-30);top=max(0,min(r[1] for r in rects)-40)
    right=min(w,max(r[0]+r[2] for r in rects)+30);bottom=min(h,max(r[1]+r[3] for r in rects)+40)
    return [left,top,right-left,bottom-top]


def set_signal(ss,p,workspace,lane,rect,event_name):
    src=ss.sources[p.source_id]
    previous=workspace['lanes'][lane].get('signal')
    sync_size=lane==0 and rect is not None and (previous is None or list(rect[2:])!=list(previous[2:]))
    if rect is not None:
        rect=list(rect)
        rect_slice(rect,src.gray.shape)
    remember(ss,p,workspace)
    roi=workspace['lanes'][lane];roi['signal']=list(rect) if rect is not None else None
    if roi.get('background_origin')!='manual':roi['background']=suggest_background(src,p,rect) if rect is not None else None
    roi['origin']='manual';roi['detected']=None
    if sync_size:match_first_lane_size(src,p,workspace)
    touch(ss,workspace);ss.roi_revision+=1
    ss.audit.add(event_name,{'protein_id':p.id,'lane':lane+1,'rect':rect,'background_mode':'automatic_adjacent'})


def set_background(ss,p,workspace,lane,rect,event_name='background_edited',automatic=False):
    src=ss.sources[p.source_id]
    if rect is not None:rect_slice(rect,src.gray.shape)
    remember(ss,p,workspace)
    roi=workspace['lanes'][lane];roi['background']=list(rect) if rect is not None else None
    roi['background_origin']='automatic' if automatic else 'manual'
    touch(ss,workspace);ss.roi_revision+=1
    ss.audit.add(event_name,{'protein_id':p.id,'lane':lane+1,'rect':rect,'background_origin':roi['background_origin']})


def edit_protein(ss,component,p,role,threshold,fraction):
    stores=ss.settings['quant_workspaces'];workspace=stores[p.id];src=ss.sources[p.source_id]
    prefix=f'seq_{ss.project_revision}_{role}_{p.id}';lane_key=prefix+'_lane'
    pending=ss.pop(prefix+'_selection',None)
    if pending:
        ss[lane_key]=pending['lane']
        if pending.get('kind'):ss[prefix+'_kind']=pending['kind']
    if st.button('自动选框',key=prefix+'_auto'):
        try:
            generated=propose_workspace(src,p,len(ss.samples),threshold,fraction,region=workspace.get('search_region'))
            match_first_lane_size(src,p,generated)
            generated['background_mode']='automatic_adjacent';generated['revision']=workspace['revision']+1
            remember(ss,p,workspace);stores[p.id]=generated;ss[prefix+'_view']=band_view(src,generated)
            ss.roi_revision+=1;ss.exported=None
            ss.audit.add('automatic_rois_generated',{'protein_id':p.id,'role':role,'threshold':threshold,'fraction':fraction})
            st.rerun()
        except ValueError as error:st.error(str(error))
    labels=[f"{i+1} · {sample['sample']}" for i,sample in enumerate(ss.samples)]
    lane=st.selectbox('内参泳道' if role=='reference' else '靶蛋白泳道',range(len(labels)),format_func=lambda i:labels[i],key=lane_key)
    show_background=st.checkbox('显示背景框',value=False,key=prefix+'_show_background')
    kind=st.radio('编辑对象',['signal','background'],format_func=lambda k:'条带框（橙色）' if k=='signal' else '背景框（蓝色）',horizontal=True,key=prefix+'_kind') if show_background else 'signal'
    tools=['move','resize','draw']
    if ss.get(prefix+'_tool') not in tools:ss[prefix+'_tool']='move'
    tool=st.radio('操作方式', tools,format_func=lambda k:{'move':'移动框','resize':'调整大小','draw':'新建 / 重画'}[k],horizontal=True,key=prefix+'_tool')
    roi=workspace['lanes'][lane];rect=roi.get('signal') if kind=='signal' else background_for_roi(src,p,workspace,roi)
    st.caption('自动选框时先统一大小；修改第 1 泳道条带框的宽高会重新同步其他现有框。每个泳道均可单独移动、调整大小。仅移动第 1 泳道条带框不会覆盖其他框的大小；删除的框不会重新出现。')
    if show_background:
        st.caption('橙色为条带，蓝色为实际用于计算的背景。手动调整背景后会固定保留，移动条带也不会覆盖；隐藏只关闭显示。背景框请避开条带。重新点击“自动选框”会重置该蛋白所有条带框和手动背景。')
        st.caption('当前泳道背景：'+('手动指定' if roi.get('background_origin')=='manual' else '自动选择'))
        if st.button('当前泳道恢复自动背景',key=prefix+'_reset_background'):
            set_background(ss,p,workspace,lane,suggest_background(src,p,roi['signal']) if roi.get('signal') else None,'background_reset_automatic',automatic=True);st.rerun()
    setter=set_signal if kind=='signal' else set_background
    controls=st.columns(3)
    if controls[0].button('复制框',disabled=rect is None,key=prefix+'_copy'):
        ss.roi_clipboard={'rect':list(rect),'kind':kind,'protein':p.name};st.success('已复制。切换泳道或蛋白后粘贴，再移动到目标位置。')
    if controls[1].button('粘贴框',disabled=not bool(ss.get('roi_clipboard')),key=prefix+'_paste'):
        try:
            setter(ss,p,workspace,lane,list(ss.roi_clipboard['rect']),'roi_pasted');st.rerun()
        except ValueError:st.error('复制的框超出当前原图范围。')
    if controls[2].button('删除框',disabled=rect is None,key=prefix+'_delete'):
        setter(ss,p,workspace,lane,None,'roi_deleted');st.rerun()
    rectangles=[];visible_rects=[];backgrounds=[]
    for i,item in enumerate(workspace['lanes']):
        entries=[('signal',item.get('signal'),'#f59e0b',str(i+1))]
        if show_background:
            bg=background_for_roi(src,p,workspace,item)
            if bg:backgrounds.append(bg)
            entries.append(('background',bg,'#0ea5e9','背景 '+str(i+1)))
        for frame_kind,frame,color,label in entries:
            if frame is None:continue
            visible_rects.append(frame)
            if i!=lane or frame_kind!=kind:
                rectangles.append({'rect':frame,'color':color,'label':label,'lane':i,'kind':frame_kind,'selectable':True})
    zoom=st.checkbox('放大条带区域',value=True,key=prefix+'_zoom')
    if st.button('视图适配所有条带',key=prefix+'_fit'):ss[prefix+'_view']=band_view(src,workspace,backgrounds)
    if prefix+'_view' not in ss:ss[prefix+'_view']=band_view(src,workspace,backgrounds)
    viewport=ss[prefix+'_view'] if zoom else [0,0,src.gray.shape[1],src.gray.shape[0]]
    if zoom and any(r[0]<viewport[0] or r[1]<viewport[1] or r[0]+r[2]>viewport[0]+viewport[2] or r[1]+r[3]>viewport[1]+viewport[3] for r in visible_rects):
        viewport=band_view(src,workspace,backgrounds);ss[prefix+'_view']=viewport
    x,y,vw,vh=viewport;preview=display_image(src).crop((x,y,x+vw,y+vh))
    context=f'{prefix}:{lane}:{kind}';h,w=src.gray.shape
    event=component(image=image_data_url(preview),raw_width=w,raw_height=h,viewport=viewport,rectangles=rectangles,
        crop=rect,angle=0,mode='crop',edit_tool=tool,lane=lane,context=context,revision=ss.roi_revision,
        crop_color='#f59e0b' if kind=='signal' else '#0ea5e9',kind=kind,aspect=0,fixed_size=None,allow_rotation=False,max_display_height=480,
        hint='所有框均可移动和调整大小；第 1 泳道条带框改变宽高时同步其他条带框。定量使用原图像素。',key=prefix+'_canvas',default=None)
    if event and event.get('revision')==ss.roi_revision and event.get('context',context)==context and event.get('nonce')!=ss.last_roi_nonce:
        ss.last_roi_nonce=event['nonce'];chosen=event.get('lane');chosen_kind=event.get('kind',kind)
        if chosen_kind not in ('signal','background') or (chosen_kind=='background' and not show_background):return
        if isinstance(chosen,int) and 0<=chosen<len(ss.samples):
            if event.get('action')=='select':
                ss[prefix+'_selection']={'lane':chosen,'kind':chosen_kind};ss.roi_revision+=1;st.rerun()
            elif 'rect' in event:
                try:
                    (set_signal if chosen_kind=='signal' else set_background)(ss,p,workspace,chosen,event['rect'],'roi_edited')
                    if chosen!=lane or chosen_kind!=kind:ss[prefix+'_selection']={'lane':chosen,'kind':chosen_kind}
                    st.rerun()
                except ValueError as error:st.error(str(error))
    zero=st.checkbox('该泳道无条带，按 0 计算',value=roi.get('no_band',False),key=f'{prefix}_zero_{lane}_{workspace["revision"]}_{ss.roi_revision}')
    if zero!=roi.get('no_band',False):
        remember(ss,p,workspace);roi['no_band']=zero;touch(ss,workspace)
        ss.audit.add('lane_no_band_flag',{'protein_id':p.id,'lane':lane+1,'value':zero})
    if roi.get('origin')=='automatic' and roi.get('detected') is False:
        if st.button('按当前框人工测量',key=prefix+'_use'):
            remember(ss,p,workspace);roi['origin']='manual';roi['detected']=None;touch(ss,workspace);ss.roi_revision+=1;st.rerun()
    nav=st.columns(2)
    def navigate(to):ss[lane_key]=to;ss.roi_revision+=1
    nav[0].button('上一泳道',disabled=lane==0,on_click=navigate,args=(max(0,lane-1),),key=prefix+'_previous')
    nav[1].button('下一泳道',disabled=lane==len(labels)-1,on_click=navigate,args=(min(len(labels)-1,lane+1),),key=prefix+'_next')
    if rect is None:st.caption('当前编辑对象没有框，可拖动创建或粘贴；缺少条带框或背景框时不能计算。')


def render_quantification(ss,component):
    raw=bio=summary=None;chart_files={}
    st.subheader('定量与归一化')
    st.caption('内参选框 → 靶蛋白选框 → 最后计算。所有自动框均可编辑。')
    proteins={p.id:p for p in ss.proteins};names={key:p.name for key,p in proteins.items()}
    if ss.get('quant_undo_project')!=ss.project_revision:
        ss.quant_undo=[];ss.quant_undo_project=ss.project_revision
    history=ss.setdefault('quant_undo',[])
    if st.button('撤销上一步',disabled=not bool(history),key='quant_undo_button'):
        item=history.pop();pid=item['protein_id']
        if pid in proteins:
            restored=copy.deepcopy(item['workspace']);restored['revision']=ss.settings['quant_workspaces'][pid]['revision']+1
            ss.settings['quant_workspaces'][pid]=restored
            for role in ['reference','target']:
                ss[f'seq_{ss.project_revision}_{role}_{pid}_view']=band_view(ss.sources[proteins[pid].source_id],restored)
            ss.roi_revision+=1;ss.exported=None
            ss.audit.add('quantification_edit_undone',{'protein_id':pid})
        st.rerun()
    st.caption('可勾选“显示背景框”查看及编辑实际背景；隐藏不影响扣除。条带和背景修改均支持撤销，最多保留 100 步。')
    stores=ss.settings.setdefault('quant_workspaces',{})
    for p in ss.proteins:
        state=stores.get(p.id)
        if not state or state.get('source_id')!=p.source_id or len(state.get('lanes',[]))!=len(ss.samples):stores[p.id]=empty_workspace(p,len(ss.samples))
        stores[p.id].pop('row_dirty',None)
        if stores[p.id].get('background_mode')!='automatic_adjacent':
            stores[p.id]['background_mode']='automatic_adjacent'
            for roi in stores[p.id]['lanes']:
                if roi.get('background') is not None:roi.setdefault('background_origin','manual')
                else:roi['background']=suggest_background(ss.sources[p.source_id],p,roi['signal']) if roi.get('signal') else None
    with st.expander('检测与测量设置',expanded=False):
        threshold=st.slider('检测阈值 / 背景噪声倍数',2.,15.,float(ss.settings.get('q_threshold',5.)),.5,key='seq_threshold')
        fraction=st.slider('自动框宽度 / 泳道间距',.4,1.,float(ss.settings.get('q_fraction',.8)),.05,key='seq_fraction')
        exclude=st.checkbox('剔除饱和条带或背景',value=ss.settings.get('exclude_saturated',True),key='seq_exclude')
        st.caption('阈值只影响下一次自动选框。定量读取原始像素，不读取亮度、对比度或反色展示。')
    ss.settings.update(q_threshold=threshold,q_fraction=fraction,exclude_saturated=exclude)
    st.markdown('**1 · 选择内参，自动选框并调整**')
    options=['']+list(proteins);old=ss.settings.get('reference_id') or ''
    choice=st.selectbox('内参 Loading Control',options,index=options.index(old) if old in options else 0,format_func=lambda k:'请选择内参' if k=='' else names[k],key=f'seq_reference_{ss.project_revision}')
    refid=choice or None;ss.settings['reference_id']=refid;ss.settings['control_condition']=None
    if refid:edit_protein(ss,component,proteins[refid],'reference',threshold,fraction)
    else:st.info('先选择内参蛋白，再点击其“自动选框”。')
    st.markdown('**2 · 选择靶蛋白，自动选框并调整**')
    targetoptions=[key for key in proteins if key!=refid]
    defaults=[key for key in ss.settings.get('target_ids',targetoptions) if key in targetoptions]
    targetids=st.multiselect('归一化靶蛋白',targetoptions,default=defaults,format_func=lambda k:names[k],key=f'seq_targets_{ss.project_revision}')
    ss.settings['target_ids']=targetids
    if targetids:
        view=st.selectbox('调整哪个靶蛋白',targetids,format_func=lambda k:names[k],key=f'seq_target_view_{ss.project_revision}')
        edit_protein(ss,component,proteins[view],'target',threshold,fraction)
    else:st.info('选择至少一个靶蛋白。多个靶蛋白可以分别调整。')
    selected=[proteins[key] for key in targetids]+([proteins[refid]] if refid else [])
    if refid and targetids and any(ss.sources[proteins[k].source_id].gray.dtype!=ss.sources[proteins[refid].source_id].gray.dtype for k in targetids):
        st.warning('靶蛋白与内参使用了不同位深的图像页（例如 16-bit 信号页与 8-bit 显示页）。强度单位可能不同，请核对原始信号页；不要直接解释该比值的绝对大小。')
    st.markdown('**3 · 最后计算条带强度与归一化**')
    overview=[]
    for item in selected:
        for i,roi in enumerate(stores[item.id]['lanes']):overview.append({'蛋白':item.name,'角色':'内参' if item.id==refid else '靶蛋白','泳道':i+1,'条带框':'已设置' if roi.get('signal') else '缺少'})
    with st.expander('计算前查看所有泳道的框'):st.dataframe(pd.DataFrame(overview),hide_index=True)
    fp=quant_fingerprint(ss.sources,selected,ss.samples,refid,stores,exclude)
    if st.button('计算条带强度与归一化',type='primary',key='q_calculate'):
        try:
            if not refid or not targetids:raise ValueError('请选择内参和至少一个靶蛋白。')
            values=calculate(ss.sources,[proteins[k] for k in targetids],proteins[refid],ss.samples,stores,exclude)
            ss.quant_result={'fingerprint':fp,'raw':values[0],'bio':values[1],'summary':values[2],'details':values[3]}
            ss.audit.add('quantification_calculated',{'fingerprint':fp,'targets':targetids,'reference':refid,'rois':{item.id:copy.deepcopy(stores[item.id]) for item in selected},'exclude_saturated':exclude})
            ss.exported=None
        except (ValueError,KeyError,IndexError) as error:st.error('计算未完成：'+str(error))
    result=ss.get('quant_result')
    if not result or result['fingerprint']!=fp:
        if result:st.info('选框或参数已改变，需重新计算；旧结果不会被导出。')
        else:st.info('框准备好后，点击最后一步的“计算条带强度与归一化”。')
        return raw,bio,summary,chart_files
    raw,bio,summary=result['raw'],result['bio'],result['summary'];details=result['details']
    ref_table=details[details.role=='reference'][['protein','lane','sample','analysis_intensity','measurement_note','analysis_status']].copy()
    ref_table.analysis_status=ref_table.analysis_status.map(status_label)
    if not refid:st.caption('当前未选择内参，仅显示条带强度。')
    if refid:st.markdown('**内参强度**');st.dataframe(ref_table.rename(columns={'protein':'内参蛋白','lane':'泳道','sample':'样品','analysis_intensity':'内参强度','measurement_note':'测量结果','analysis_status':'状态'}),hide_index=True)
    labels={'protein':'靶蛋白','lane':'泳道','sample':'样品','analysis_intensity':'靶蛋白强度','reference_intensity':'内参强度','normalized_ratio':'靶蛋白 / 内参','measurement_note':'测量结果','analysis_status':'状态'}
    table=raw[list(labels)].copy().rename(columns=labels);table['状态']=table['状态'].map(status_label)
    table['靶蛋白 / 内参']=table['靶蛋白 / 内参'].map(lambda v:'无法计算' if pd.isna(v) else f'{v:.6g}')
    if not refid:table=table.drop(columns=['内参强度','靶蛋白 / 内参'])
    st.markdown('**归一化结果（可复制）**' if refid else '**全部泳道强度（可复制）**');st.dataframe(table,hide_index=True)
    excel=export_excel(raw,bio,summary,details)
    downloads=st.columns(3)
    downloads[0].download_button('下载归一化数值 CSV',safe_csv(raw),file_name='normalization.csv',mime='text/csv')
    downloads[1].download_button('下载定量 Excel',excel,file_name='analysis.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    downloads[2].download_button('下载全部强度 CSV（含内参）',safe_csv(details),file_name='all_intensities.csv',mime='text/csv')
    with st.expander('背景、原始积分及复孔统计'):
        st.dataframe(details,hide_index=True)
        if summary.empty:st.caption('填入条件和重复编号后可计算均值及 SD。')
        else:st.dataframe(summary,hide_index=True)
    if refid and raw.normalized_ratio.isna().any():st.warning('部分泳道无法归一化，请查看内参和状态。缺失比值不会被画成 0。')
    st.markdown('**柱状图**')
    chartid=st.selectbox('绘图蛋白',targetids,format_func=lambda k:names[k],key='q_chart_protein')
    chartmode=st.radio('柱状图内容',['逐泳道比值','分组均值 ± SD'],horizontal=True,key='q_chart_mode')
    grouped=chartmode=='分组均值 ± SD'
    try:
        if grouped and bio.empty:raise ValueError('分组图需要条件、生物学重复和技术重复编号。')
        chart_files,caption=normalization_chart(raw,bio,chartid,names[refid],grouped=grouped,dpi=ss.settings['dpi'],font_family=ss.settings['protein_font_family'],font_pt=ss.settings['protein_font_pt'],font_style=ss.settings.get('protein_font_style','regular'))
        st.image(chart_files['normalization.png'],caption=caption)
        st.download_button('下载柱状图 PNG',chart_files['normalization.png'],file_name='normalization.png',mime='image/png')
        st.download_button('下载柱状图 PDF',chart_files['normalization.pdf'],file_name='normalization.pdf',mime='application/pdf')
    except ValueError as error:st.info(str(error))
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in {'normalization.csv':safe_csv(raw),'all_intensities.csv':safe_csv(details),'analysis.xlsx':excel,**chart_files}.items():archive.writestr(name,data)
    st.download_button('下载本次定量结果包',bundle.getvalue(),file_name='quantification-results.zip',mime='application/zip')
    return raw,bio,summary,chart_files
