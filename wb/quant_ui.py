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
    names={'invalid_or_nonpositive_reference':'Control has no valid positive intensity; normalization is unavailable','missing_signal_or_background':'Missing band region or background estimate',
           'background_overlaps_signal':'Band and background regions overlap','saturated_signal_or_background':'Band or background contains saturated pixels',
           'pixels_outside_sensor_range':'Pixels outside the sensor range','membrane_pairing_mismatch':'Target and control membrane IDs differ'}
    parts=[names.get(v,v) for v in str(value).split(';') if v!='ok']
    return '; '.join(parts) if parts else 'Calculated'


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
    if st.button('Auto-select regions',key=prefix+'_auto'):
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
    lane=st.selectbox('Control lane' if role=='reference' else 'Target lane',range(len(labels)),format_func=lambda i:labels[i],key=lane_key)
    show_background=st.checkbox('Show background regions',value=False,key=prefix+'_show_background')
    kind=st.radio('Edit region',['signal','background'],format_func=lambda k:'Band (orange)' if k=='signal' else 'Background (blue)',horizontal=True,key=prefix+'_kind') if show_background else 'signal'
    tools=['move','resize','draw']
    if ss.get(prefix+'_tool') not in tools:ss[prefix+'_tool']='move'
    tool=st.radio('Edit tool', tools,format_func=lambda k:{'move':'Move','resize':'Resize','draw':'Draw / redraw'}[k],horizontal=True,key=prefix+'_tool')
    roi=workspace['lanes'][lane];rect=roi.get('signal') if kind=='signal' else background_for_roi(src,p,workspace,roi)
    st.caption("Auto-selection starts with equal sizes. Resizing lane 1's band region resizes the other existing band regions. All lanes can then be moved or resized individually. Moving lane 1 alone preserves custom sizes; deleted regions stay deleted.")
    if show_background:
        st.caption('Orange: band; blue: background used in calculation. Manual backgrounds remain fixed when bands move. Hiding affects display only. Avoid bands in background regions. Auto-select resets all regions for this protein, including manual backgrounds.')
        st.caption('Current lane background: '+('Manual' if roi.get('background_origin')=='manual' else 'Automatic'))
        if st.button('Reset this lane to automatic background',key=prefix+'_reset_background'):
            set_background(ss,p,workspace,lane,suggest_background(src,p,roi['signal']) if roi.get('signal') else None,'background_reset_automatic',automatic=True);st.rerun()
    setter=set_signal if kind=='signal' else set_background
    controls=st.columns(3)
    if controls[0].button('Copy region',disabled=rect is None,key=prefix+'_copy'):
        ss.roi_clipboard={'rect':list(rect),'kind':kind,'protein':p.name};st.success('Copied. Switch lane or protein, paste, then move the region to the intended position.')
    if controls[1].button('Paste region',disabled=not bool(ss.get('roi_clipboard')),key=prefix+'_paste'):
        try:
            setter(ss,p,workspace,lane,list(ss.roi_clipboard['rect']),'roi_pasted');st.rerun()
        except ValueError:st.error('Copied region exceeds the source image.')
    if controls[2].button('Delete region',disabled=rect is None,key=prefix+'_delete'):
        setter(ss,p,workspace,lane,None,'roi_deleted');st.rerun()
    rectangles=[];visible_rects=[];backgrounds=[]
    for i,item in enumerate(workspace['lanes']):
        entries=[('signal',item.get('signal'),'#f59e0b',str(i+1))]
        if show_background:
            bg=background_for_roi(src,p,workspace,item)
            if bg:backgrounds.append(bg)
            entries.append(('background',bg,'#0ea5e9','BG '+str(i+1)))
        for frame_kind,frame,color,label in entries:
            if frame is None:continue
            visible_rects.append(frame)
            if i!=lane or frame_kind!=kind:
                rectangles.append({'rect':frame,'color':color,'label':label,'lane':i,'kind':frame_kind,'selectable':True})
    zoom=st.checkbox('Zoom to band area',value=True,key=prefix+'_zoom')
    if st.button('Fit all regions',key=prefix+'_fit'):ss[prefix+'_view']=band_view(src,workspace,backgrounds)
    if prefix+'_view' not in ss:ss[prefix+'_view']=band_view(src,workspace,backgrounds)
    viewport=ss[prefix+'_view'] if zoom else [0,0,src.gray.shape[1],src.gray.shape[0]]
    if zoom and any(r[0]<viewport[0] or r[1]<viewport[1] or r[0]+r[2]>viewport[0]+viewport[2] or r[1]+r[3]>viewport[1]+viewport[3] for r in visible_rects):
        viewport=band_view(src,workspace,backgrounds);ss[prefix+'_view']=viewport
    x,y,vw,vh=viewport;preview=display_image(src).crop((x,y,x+vw,y+vh))
    context=f'{prefix}:{lane}:{kind}';h,w=src.gray.shape
    event=component(image=image_data_url(preview),raw_width=w,raw_height=h,viewport=viewport,rectangles=rectangles,
        crop=rect,angle=0,mode='crop',edit_tool=tool,lane=lane,context=context,revision=ss.roi_revision,
        crop_color='#f59e0b' if kind=='signal' else '#0ea5e9',kind=kind,aspect=0,fixed_size=None,allow_rotation=False,max_display_height=480,
        hint="Move or resize any region. Resizing lane 1's band synchronizes other band sizes. Measurements use native pixels.",key=prefix+'_canvas',default=None)
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
    zero=st.checkbox('No visible band in this lane: assign zero',value=roi.get('no_band',False),key=f'{prefix}_zero_{lane}_{workspace["revision"]}_{ss.roi_revision}')
    if zero!=roi.get('no_band',False):
        remember(ss,p,workspace);roi['no_band']=zero;touch(ss,workspace)
        ss.audit.add('lane_no_band_flag',{'protein_id':p.id,'lane':lane+1,'value':zero})
    if roi.get('origin')=='automatic' and roi.get('detected') is False:
        if st.button('Measure current region manually',key=prefix+'_use'):
            remember(ss,p,workspace);roi['origin']='manual';roi['detected']=None;touch(ss,workspace);ss.roi_revision+=1;st.rerun()
    nav=st.columns(2)
    def navigate(to):ss[lane_key]=to;ss.roi_revision+=1
    nav[0].button('Previous lane',disabled=lane==0,on_click=navigate,args=(max(0,lane-1),),key=prefix+'_previous')
    nav[1].button('Next lane',disabled=lane==len(labels)-1,on_click=navigate,args=(min(len(labels)-1,lane+1),),key=prefix+'_next')
    if rect is None:st.caption('No region for the selected object. Draw or paste one. Both band and background regions are required for measurement.')


def render_quantification(ss,component):
    raw=bio=summary=None;chart_files={}
    st.subheader('Quantification and normalization')
    st.caption('Control regions → target regions → calculate. All suggested regions can be edited.')
    proteins={p.id:p for p in ss.proteins};names={key:p.name for key,p in proteins.items()}
    if ss.get('quant_undo_project')!=ss.project_revision:
        ss.quant_undo=[];ss.quant_undo_project=ss.project_revision
    history=ss.setdefault('quant_undo',[])
    if st.button('Undo last edit',disabled=not bool(history),key='quant_undo_button'):
        item=history.pop();pid=item['protein_id']
        if pid in proteins:
            restored=copy.deepcopy(item['workspace']);restored['revision']=ss.settings['quant_workspaces'][pid]['revision']+1
            ss.settings['quant_workspaces'][pid]=restored
            for role in ['reference','target']:
                ss[f'seq_{ss.project_revision}_{role}_{pid}_view']=band_view(ss.sources[proteins[pid].source_id],restored)
            ss.roi_revision+=1;ss.exported=None
            ss.audit.add('quantification_edit_undone',{'protein_id':pid})
        st.rerun()
    st.caption('Show background regions to review/edit the background used for measurement. Hiding does not disable subtraction. Band and background edits support up to 100 undo steps.')
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
    with st.expander('Detection and measurement settings',expanded=False):
        threshold=st.slider('Detection threshold / background-noise multiple',2.,15.,float(ss.settings.get('q_threshold',5.)),.5,key='seq_threshold')
        fraction=st.slider('Auto region width / lane spacing',.4,1.,float(ss.settings.get('q_fraction',.8)),.05,key='seq_fraction')
        exclude=st.checkbox('Exclude saturated bands or backgrounds',value=ss.settings.get('exclude_saturated',True),key='seq_exclude')
        st.caption('Threshold affects the next auto-selection only. Quantification uses native pixels, not display brightness, contrast or inversion.')
    ss.settings.update(q_threshold=threshold,q_fraction=fraction,exclude_saturated=exclude)
    st.markdown('**1 · Select a loading control and review regions**')
    options=['']+list(proteins);old=ss.settings.get('reference_id') or ''
    choice=st.selectbox('Loading control',options,index=options.index(old) if old in options else 0,format_func=lambda k:'Select a loading control' if k=='' else names[k],key=f'seq_reference_{ss.project_revision}')
    refid=choice or None;ss.settings['reference_id']=refid;ss.settings['control_condition']=None
    if refid:edit_protein(ss,component,proteins[refid],'reference',threshold,fraction)
    else:st.info('Select a control protein, then click "Auto-select regions".')
    st.markdown('**2 · Select targets and review regions**')
    targetoptions=[key for key in proteins if key!=refid]
    defaults=[key for key in ss.settings.get('target_ids',targetoptions) if key in targetoptions]
    targetids=st.multiselect('Targets to normalize',targetoptions,default=defaults,format_func=lambda k:names[k],key=f'seq_targets_{ss.project_revision}')
    ss.settings['target_ids']=targetids
    if targetids:
        view=st.selectbox('Target protein to edit',targetids,format_func=lambda k:names[k],key=f'seq_target_view_{ss.project_revision}')
        edit_protein(ss,component,proteins[view],'target',threshold,fraction)
    else:st.info('Select at least one target. Each target can be edited separately.')
    selected=[proteins[key] for key in targetids]+([proteins[refid]] if refid else [])
    if refid and targetids and any(ss.sources[proteins[k].source_id].gray.dtype!=ss.sources[proteins[refid].source_id].gray.dtype for k in targetids):
        st.warning("Target and control use different image bit depths (e.g. 16-bit signal vs 8-bit display). Units may differ. Verify source pages; do not interpret the ratio's absolute magnitude directly.")
    st.markdown('**3 · Calculate intensities and normalization**')
    overview=[]
    for item in selected:
        for i,roi in enumerate(stores[item.id]['lanes']):overview.append({'Protein':item.name,'Role':'Control' if item.id==refid else 'Target','Lane':i+1,'Band region':'Set' if roi.get('signal') else 'Missing'})
    with st.expander('Review all lane regions before calculation'):st.dataframe(pd.DataFrame(overview),hide_index=True)
    fp=quant_fingerprint(ss.sources,selected,ss.samples,refid,stores,exclude)
    if st.button('Calculate intensities and normalization',type='primary',key='q_calculate'):
        try:
            if not refid or not targetids:raise ValueError('Select a loading control and at least one target.')
            values=calculate(ss.sources,[proteins[k] for k in targetids],proteins[refid],ss.samples,stores,exclude)
            ss.quant_result={'fingerprint':fp,'raw':values[0],'bio':values[1],'summary':values[2],'details':values[3]}
            ss.audit.add('quantification_calculated',{'fingerprint':fp,'targets':targetids,'reference':refid,'rois':{item.id:copy.deepcopy(stores[item.id]) for item in selected},'exclude_saturated':exclude})
            ss.exported=None
        except (ValueError,KeyError,IndexError) as error:st.error('Calculation incomplete: '+str(error))
    result=ss.get('quant_result')
    if not result or result['fingerprint']!=fp:
        if result:st.info('Regions or parameters changed. Recalculate before export; previous results are stale.')
        else:st.info('Review the regions, then click "Calculate intensities and normalization".')
        return raw,bio,summary,chart_files
    raw,bio,summary=result['raw'],result['bio'],result['summary'];details=result['details']
    ref_table=details[details.role=='reference'][['protein','lane','sample','analysis_intensity','measurement_note','analysis_status']].copy()
    ref_table.analysis_status=ref_table.analysis_status.map(status_label)
    if not refid:st.caption('No control selected; showing band intensities only.')
    if refid:st.markdown('**Control intensities**');st.dataframe(ref_table.rename(columns={'protein':'Control protein','lane':'Lane','sample':'Sample','analysis_intensity':'Control intensity','measurement_note':'Measurement','analysis_status':'Status'}),hide_index=True)
    labels={'protein':'Target','lane':'Lane','sample':'Sample','analysis_intensity':'Target intensity','reference_intensity':'Control intensity','normalized_ratio':'Target / control','measurement_note':'Measurement','analysis_status':'Status'}
    table=raw[list(labels)].copy().rename(columns=labels);table['Status']=table['Status'].map(status_label)
    table['Target / control']=table['Target / control'].map(lambda v:'Unavailable' if pd.isna(v) else f'{v:.6g}')
    if not refid:table=table.drop(columns=['Control intensity','Target / control'])
    st.markdown('**Normalization results (copyable)**' if refid else '**All lane intensities (copyable)**');st.dataframe(table,hide_index=True)
    excel=export_excel(raw,bio,summary,details)
    downloads=st.columns(3)
    downloads[0].download_button('Download normalization CSV',safe_csv(raw),file_name='normalization.csv',mime='text/csv')
    downloads[1].download_button('Download quantification Excel',excel,file_name='analysis.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    downloads[2].download_button('Download all intensities CSV (includes control)',safe_csv(details),file_name='all_intensities.csv',mime='text/csv')
    with st.expander('Background, raw integrals and replicate statistics'):
        st.dataframe(details,hide_index=True)
        if summary.empty:st.caption('Enter conditions and replicate IDs to calculate means and SD.')
        else:st.dataframe(summary,hide_index=True)
    if refid and raw.normalized_ratio.isna().any():st.warning('Some lanes cannot be normalized. Check controls and status. Missing ratios are not plotted as zero.')
    st.markdown('**Bar chart**')
    chartid=st.selectbox('Protein to plot',targetids,format_func=lambda k:names[k],key='q_chart_protein')
    chartmode=st.radio('Chart content',['Per-lane ratios','Group mean ± SD'],horizontal=True,key='q_chart_mode')
    grouped=chartmode=='Group mean ± SD'
    try:
        if grouped and bio.empty:raise ValueError('Grouped charts require condition, biological and technical replicate IDs.')
        chart_files,caption=normalization_chart(raw,bio,chartid,names[refid],grouped=grouped,dpi=ss.settings['dpi'],font_family=ss.settings['protein_font_family'],font_pt=ss.settings['protein_font_pt'],font_style=ss.settings.get('protein_font_style','regular'))
        st.image(chart_files['normalization.png'],caption=caption)
        st.download_button('Download chart PNG',chart_files['normalization.png'],file_name='normalization.png',mime='image/png')
        st.download_button('Download chart PDF',chart_files['normalization.pdf'],file_name='normalization.pdf',mime='application/pdf')
    except ValueError as error:st.info(str(error))
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in {'normalization.csv':safe_csv(raw),'all_intensities.csv':safe_csv(details),'analysis.xlsx':excel,**chart_files}.items():archive.writestr(name,data)
    st.download_button('Download quantification bundle',bundle.getvalue(),file_name='quantification-results.zip',mime='application/zip')
    return raw,bio,summary,chart_files
