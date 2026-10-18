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
st.caption('Labels support α, β and γ; superscript: Pfkl^{S775A/S775A}; subscript: IL_{1}.')
st.caption('Native-pixel quantification · Multi-protein figures · Lane-matched loading controls · Original files and processing history')
with st.sidebar:
    st.subheader('Project and source images')
    uploads=st.file_uploader('Upload TIFF / PNG / JPG',type=['tif','tiff','png','jpg','jpeg'],accept_multiple_files=True,key='uploads_'+str(ss.project_revision))
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
    if st.button('Load synthetic demo',disabled=bool(ss.sources)):
        demo();st.rerun()
    project=st.file_uploader('Restore project ZIP',type=['zip'],key='restore_upload')
    if project and st.button('Restore project'):
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
    st.caption('Local runs process files on your computer. Hosted runs send uploads to the server. Download your project before refreshing or closing; session data may be lost.')
    st.caption('Processing history records actions within this tool; it is not a tamper-proof signature or verification of experimental authenticity.')

if not ss.sources:
    st.info('Upload source images or load the demo. Each image can contain multiple proteins, combined into one figure.')
    st.stop()

if ss.get('is_demo',False):
    st.warning('Synthetic demo: Target A, Target B and GAPDH are simulated data. Uploading your images clears the demo, or use the button below to start a blank experiment.')
    if st.button('Clear demo and start an experiment'):
        ss.sources={};ss.proteins=[];ss.loaded=set();ss.is_demo=False;ss.settings['crop_template']=None
        ss.settings['reference_id']=None;ss.settings['control_condition']=None
        ss.project_revision+=1;ss.roi_revision+=1;invalidate()
        ss.audit.add('synthetic_demo_cleared',{});st.rerun()

preferred_sources=preferred_source_ids(ss.sources)
st.caption(f"① Upload images → ② Define samples → ③ Add proteins → ④ Review regions → ⑤ Normalize and export · {len(preferred_sources)} file(s), {len(ss.proteins)} protein(s)")
tabs=st.tabs(['1 · Samples','2 · Add / manage proteins','3 · Images and cropping','4 · Quantification and normalization','5 · Figure and export','6 · Save project'])
with tabs[0]:
    st.subheader('Samples and replicates')
    n=st.number_input('Number of lanes',min_value=1,max_value=40,value=len(ss.samples))
    if n!=len(ss.samples):
        ss.samples=(ss.samples+initial_samples(n)[len(ss.samples):])[:n]
        for p in ss.proteins:
            fresh=default_protein(ss.sources[p.source_id],n)
            p.signal=fresh.signal;p.background=fresh.background
        ss.audit.add('lane_count_changed',{'lane_count':n,'roi_reset':True});ss.roi_revision+=1;invalidate()
    st.caption('Enter each lane label once, e.g. un, 15, 30, 60, 120. Use LPS as the optional group heading, or leave the group blank.')
    for sample in ss.samples:
        if sample.get('figure_label'):sample['sample']=sample.pop('figure_label')
        else:sample.pop('figure_label',None)
        sample.setdefault('figure_group','')
    table=pd.DataFrame([{k:sample.get(k,'') for k in ['sample','figure_group']} for sample in ss.samples])
    edited=st.data_editor(table,hide_index=True,num_rows='fixed',key=f'sample_table_{ss.project_revision}_{n}',column_config={
        'sample':st.column_config.TextColumn('Lane label'),'figure_group':st.column_config.TextColumn('Group heading (optional)')})
    for original,changed in zip(ss.samples,edited.fillna('').astype(str).to_dict('records')):original.update(changed)
    with st.expander('Replicate settings (for mean and SD)',expanded=False):
        stats=pd.DataFrame([{k:sample.get(k,'') for k in ['condition','bio_rep','tech_rep']} for sample in ss.samples])
        changed=st.data_editor(stats,hide_index=True,key=f'replicates_{ss.project_revision}_{n}',column_config={
            'condition':st.column_config.TextColumn('Condition'),'bio_rep':st.column_config.TextColumn('Biological replicate'),
            'tech_rep':st.column_config.TextColumn('Technical replicate')})
        for original,row in zip(ss.samples,changed.fillna('').astype(str).to_dict('records')):original.update(row)

with tabs[1]:
    st.subheader('Add proteins to analyze')
    st.write('One protein entry produces one strip in the final figure. Entries are created only with "Add protein", not inferred from filenames.')
    source_labels={k:f'{v.name} / page {v.page+1} · {v.gray.dtype.itemsize*8}-bit'+(' · palette display page' if 'palette' in v.conversion else '') for k,v in ss.sources.items()}
    show_all_pages=False
    if len(ss.sources)>len(preferred_sources):
        st.caption('One page per file is shown by default, preferring native 16-bit grayscale. Other pages and original files are retained.')
        with st.expander('Other image pages (optional)'):
            show_all_pages=st.checkbox('Show all image pages',key='show_all_source_pages')
    source_options=list(ss.sources) if show_all_pages else preferred_sources
    # Advanced page switching still starts with the preferred native page.
    if ss.get('new_protein_source') not in source_options:
        ss['new_protein_source']=preferred_sources[0]
    with st.form('add_protein_'+str(ss.project_revision),clear_on_submit=True):
        left,right=st.columns(2)
        new_name=left.text_input('Protein name',placeholder='Enter the protein you measured',key='new_protein_name')
        src_id=right.selectbox('Source image',source_options,format_func=lambda k:source_labels[k],key='new_protein_source')
        new_mw=left.text_input('Molecular weight (optional)',placeholder='e.g. 50 kDa',key='new_protein_mw')
        new_membrane=right.text_input('Membrane ID (for normalization)',placeholder='Use the same ID for proteins from the same membrane',key='new_protein_membrane')
        add=st.form_submit_button('Add protein',type='primary')
    if add:
        if not new_name.strip():st.error('Enter a protein name before adding it.')
        else:
            new=default_protein(ss.sources[src_id],len(ss.samples),new_name.strip())
            new.molecular_weight=new_mw.strip();new.membrane_id=new_membrane.strip()
            ss.proteins.append(new);ss.audit.add('protein_added',{'protein_id':new.id,'name':new.name,'source_id':new.source_id})
            ss.roi_revision+=1;invalidate();st.rerun()
    st.divider()
    st.subheader(f'Added proteins ({len(ss.proteins)})')
    st.caption('Edit the name, molecular weight, membrane ID or source image, or delete the entry. Source images and samples remain even if all proteins are removed.')
    if not ss.proteins:
        st.info('No proteins yet. Enter a name, select a source image and click "Add protein".')
    for i,item in enumerate(list(ss.proteins)):
        with st.container(border=True):
            title,edit_col,delete_col=st.columns([6,1,1])
            title.markdown(f'**{i+1}. {item.name}**')
            title.caption(f"Source: {source_labels[item.source_id]} · MW: {item.molecular_weight or 'not set'} · Membrane: {item.membrane_id or 'not set'}")
            editing_key='editing_'+item.id
            if edit_col.button('Edit',key='edit_'+item.id):
                ss[editing_key]=not ss.get(editing_key,False)
            if delete_col.button('Delete',key='delete_'+item.id):
                ss.proteins=[existing for existing in ss.proteins if existing.id!=item.id]
                if ss.settings['reference_id']==item.id:ss.settings['reference_id']=None
                ss.audit.add('protein_deleted',{'protein_id':item.id,'name':item.name})
                ss.roi_revision+=1;invalidate();st.rerun()
            if ss.get(editing_key,False):
                with st.form('edit_form_'+item.id):
                    c1,c2=st.columns(2)
                    name=c1.text_input('Protein name',value=item.name,key='edit_name_'+item.id)
                    mw=c2.text_input('Molecular weight (optional)',value=item.molecular_weight,key='edit_mw_'+item.id)
                    membrane=c1.text_input('Membrane ID',value=item.membrane_id,key='edit_membrane_'+item.id)
                    edit_options=list(source_options)
                    if item.source_id not in edit_options:edit_options.append(item.source_id)
                    source_id=c2.selectbox('Source image',edit_options,index=edit_options.index(item.source_id),format_func=lambda k:source_labels[k],key='edit_source_'+item.id)
                    save=st.form_submit_button('Save changes',type='primary')
                if save:
                    if not name.strip():st.error('Protein name cannot be empty.')
                    else:
                        if source_id!=item.source_id:
                            fresh=default_protein(ss.sources[source_id],len(ss.samples),name.strip())
                            item.crop=fresh.crop;item.signal=fresh.signal;item.background=fresh.background;item.crop_angle=0.;item.crop_selected=False;item.segments=[]
                            item.markers=[];item.manual_lines=[];item.segments=[]
                        item.name=name.strip();item.molecular_weight=mw.strip();item.membrane_id=membrane.strip();item.source_id=source_id
                        ss[editing_key]=False;ss.roi_revision+=1;invalidate()
                        ss.audit.add('protein_edited',{'protein_id':item.id,'name':item.name,'source_id':item.source_id});st.rerun()
            a,b,_=st.columns([1,1,6])
            if a.button('Move up',key='up_'+item.id,disabled=i==0):
                ss.proteins[i-1],ss.proteins[i]=ss.proteins[i],ss.proteins[i-1];ss.roi_revision+=1;invalidate();st.rerun()
            if b.button('Move down',key='down_'+item.id,disabled=i==len(ss.proteins)-1):
                ss.proteins[i+1],ss.proteins[i]=ss.proteins[i],ss.proteins[i+1];ss.roi_revision+=1;invalidate();st.rerun()
    st.caption('Verify membrane IDs yourself. The tool does not identify proteins or assume GAPDH is the loading control. Select your control in Quantification and normalization.')

if not ss.proteins:
    with tabs[2]:st.info('Add a protein under "2 · Add / manage proteins" before drawing regions.')
    with tabs[3]:st.info('Add the proteins you measured before selecting a control and quantifying.')
    with tabs[4]:st.info('The multi-protein figure will appear here after you add proteins.')
    with tabs[5]:
        checkpoint()
        st.download_button('Save source images and samples',project_bytes(ss.sources,[],ss.samples,ss.audit,ss.settings),file_name='wb-project.zip',mime='application/zip')
    st.stop()
protein_names=[p.name for p in ss.proteins]
with tabs[2]:
    p_index=st.selectbox('Protein to edit',range(len(ss.proteins)),format_func=lambda i:protein_names[i],key='active_protein_'+str(ss.project_revision))
    p=ss.proteins[p_index];src=ss.sources[p.source_id];h,w=src.gray.shape
    st.caption(f"Source {src.name} · page {src.page+1} · {w} × {h} · {src.gray.dtype} · {src.conversion}")
    if src.name.lower().endswith(('.jpg','.jpeg')):
        st.warning('JPEG is lossy and cannot recover original acquisition intensities. Prefer original lossless TIFF images.')
    operation='crop'
    st.caption('Edit band and background measurement regions in the Quantification and normalization tab.')
    src.display_invert=st.checkbox('Invert display (black / white)',value=src.display_invert,key='invert_'+src.id)
    st.caption('Display inversion applies to previews and figure exports from this image. Native pixels and quantification are unchanged.')
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
        if size_locked:st.info(f'Fixed crop size: {template[0]} × {template[1]} pixels. Subsequent crops can be moved and rotated.')
        elif template:st.info("This strip defines the size template. You can still change its width and height; confirming synchronizes the other proteins' crop sizes.")
        else:st.info('Adjust the first crop width and height, then confirm. Subsequent crops use the same fixed size.')
        padding=st.number_input('Background padding / source pixels',0,30,3,key='padding_'+p.id)
        if st.button('Detect band rows in source image',type='primary',key='detect_'+p.id):
            with st.spinner('Detecting band positions…'):
                ss['band_candidates_'+src.id]=detect_band_rows(src,padding=int(padding))
            ss.pop('candidate_selection_'+p.id,None)
            ss.audit.add('band_rows_suggested',{'source_id':src.id,'candidates':ss['band_candidates_'+src.id]})
        suggestions=ss.get('band_candidates_'+src.id)
        if suggestions is not None:
            if not suggestions:st.info('No clear candidate bands found. Draw a crop on the source image.')
            else:
                choice=st.selectbox('Candidate row to crop',range(len(suggestions)),format_func=lambda i:f"Row {i+1} · y={suggestions[i]['source_y']}",key='candidate_'+p.id)
                candidate=suggestions[choice]
                token=(src.id,choice,tuple(candidate['rect']),candidate.get('padding',3))
                if ss.get('candidate_selection_'+p.id)!=token:
                    ss['candidate_selection_'+p.id]=token
                    rect=list(candidate.get('axis_rect',candidate['rect']))
                    if size_locked:rect=fixed_crop(rect,template,src.gray.shape)
                    draft={'rect':rect,'angle':0.,'polarity':candidate['polarity']}
                    ss[draft_key]=draft;ss.roi_revision+=1
                st.caption(f"The draft region is shown below. Estimated tilt: {candidate['angle']:g}°. Rotation and cropping are not applied automatically.")
                if st.button('Use estimated tilt (optional)',key='suggested_rotation_'+p.id):
                    rect=fixed_crop(candidate['rect'],template,src.gray.shape,candidate['angle']) if size_locked else list(candidate['rect'])
                    draft={'rect':rect,'angle':candidate['angle'],'polarity':candidate['polarity']}
                    ss[draft_key]=draft;ss.roi_revision+=1;st.rerun()
        if draft:
            angle=st.slider('Rotate draft crop / °',-45.,45.,float(draft['angle']),.1,key=f'angle_{p.id}_{ss.roi_revision}')
            if angle!=draft['angle']:
                try:validate_crop(draft['rect'],angle,src.gray.shape);draft['angle']=angle
                except ValueError as error:st.error(str(error))
        st.caption('This changes only the draft crop. Then click "Confirm crop" below the image. All figure strips retain equal width and height.')
    with st.expander('Saturation and native signal settings',expanded=False):
        ca,cb,cc=st.columns(3)
        p.polarity=ca.selectbox('Signal polarity',['dark','light'],index=['dark','light'].index(p.polarity),format_func=lambda x:polarity_label(src,x),key=f'polarity_encoding_{p.id}_{ss.roi_revision}_{src.native_inverted}_{src.display_invert}')
        src.floor=cb.number_input('Sensor minimum',value=float(src.floor),key=f'floor_{src.id}')
        src.ceiling=cc.number_input('Sensor maximum',value=float(src.ceiling),key=f'ceiling_{src.id}')
        if src.native_inverted:st.caption('Inverse TIFF grayscale encoding detected. Options describe the displayed band color and are mapped to native pixel polarity automatically.')
        st.caption('Choose the displayed band color. Inverting the display updates option labels, not native pixels or existing measurement polarity. Default 16-bit maximum: 65535; unscaled 12-bit: 4095.')
        try:
            mask=saturation(src,p.polarity)
            st.caption(f"Potential saturation: {int(mask.sum()):,} pixels ({mask.mean()*100:.4f}%). Threshold checks do not establish detector linearity.")
        except ValueError as error:st.error(str(error))
        overlay=st.checkbox('Highlight saturated pixels in red (screen only)',value=False,key='overlay_'+p.id)
    with st.expander('Whole-image display adjustments',expanded=False):
        c1,c2=st.columns(2)
        src.display_black=c1.number_input('Display black point',value=float(src.display_black),key=f'black_{src.id}')
        src.display_white=c2.number_input('Display white point',value=float(src.display_white),key=f'white_{src.id}')
        src.brightness=c1.slider('Brightness',-.5,.5,float(src.brightness),.01,key=f'brightness_{src.id}')
        src.contrast=c2.slider('Contrast',.1,3.0,float(src.contrast),.05,key=f'contrast_{src.id}')
        st.caption('Linear adjustments apply uniformly to the whole source image and all proteins from it. Quantification does not use the display image.')
    try:
        mask=saturation(src,p.polarity)
        mode='crop';lane=0;rectangles=[]
        st.caption('Drag inside to move; use the top handle to rotate. With fixed size, click outside to reposition. The first crop can be resized before confirmation.')
        event=roi_component(image=image_data_url(display_image(src,mask if overlay else None)),raw_width=w,raw_height=h,
                            rectangles=rectangles,mode=mode,lane=lane,revision=ss.roi_revision,
                            crop=draft['rect'] if operation=='crop' and draft else None,angle=draft['angle'] if draft else 0.,
                            aspect=0,fixed_size=template if operation=='crop' and size_locked else None,
                            hint=('Fixed size: drag inside to move, use the top handle to rotate, or click outside to reposition.' if size_locked else 'Drag inside to move; use the top handle to rotate and edges or corners to resize.') if operation=='crop' else "Draw the current lane's band or background region. Separator mode uses the left edge.",key='roi_'+str(ss.project_revision)+'_'+p.id+'_'+operation,default=None)
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
        st.caption(f"Draft crop: {draft['rect'][2]} × {draft['rect'][3]} source pixels · {draft['angle']:g}°")
        if st.button('Confirm crop',type='primary',key='confirm_crop_'+p.id):
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
        with st.expander('Confirmed crop (for final figure)',expanded=False):
            try:st.image(extract_strip(src,p))
            except ValueError as error:st.error(str(error))
    with st.expander('Markers, noncontiguous lanes and separators'):
        st.caption('Retain necessary background and original markers when cropping. kDa text does not create or replace actual marker bands. Noncontiguous segments receive visible join separators.')
        st.caption('Position marker labels in the Figure tab. Existing multi-marker labels use source-image y coordinates mapped into the crop.')
        segments=st.data_editor(pd.DataFrame(p.segments,columns=['x','width']),num_rows='dynamic',hide_index=True,key='segments_'+p.id)
        p.segments=[[int(row.x),int(row.width)] for row in segments.itertuples() if pd.notna(row.x) and pd.notna(row.width)]
        st.caption('Leave blank for a continuous crop. Segments must be within the crop, ordered from left to right.')
        lines=st.text_input('Manual separator x coordinates in source pixels, comma-separated',value=','.join(map(str,p.manual_lines)),key='lines_'+p.id+'_'+str(ss.roi_revision))
        try:p.manual_lines=[int(x.strip()) for x in lines.split(',') if x.strip()]
        except ValueError:st.error('Separator coordinates must be integers.')


for field in ['dpi','lane_font_pt','protein_font_pt','marker_font_pt','lane_font_family','protein_font_family','marker_font_family','lane_font_style','protein_font_style','marker_font_style']:
    widget_key='layout_'+field+'_'+str(ss.project_revision)
    if widget_key in ss:ss.settings[field]=ss[widget_key]
ss.settings['font_family']=ss.settings['protein_font_family'];ss.settings['font_pt']=ss.settings['protein_font_pt']
chart_files={}
with tabs[3]:
    raw,bio,summary,chart_files=render_quantification(ss,roi_component)

ref_names={item.id:item.name for item in ss.proteins}
with tabs[4]:
    st.subheader('Multi-protein figure')
    controls=st.columns(2)
    ss.settings['dpi']=controls[0].selectbox('DPI',[300,600],index=[300,600].index(ss.settings['dpi']),key='layout_dpi_'+str(ss.project_revision))
    ss.settings['gap_mm']=controls[1].number_input('Strip spacing / mm',.2,15.,float(ss.settings['gap_mm']),.2)
    families=list(FONT_FILES)
    text_columns=st.columns(3)
    for col,role,label in zip(text_columns,['lane','protein','marker'],['Lane labels','Protein names','Marker']):
        col.markdown('**'+label+' annotations**')
        family=role+'_font_family';size=role+'_font_pt'
        ss.settings[family]=col.selectbox(label+' font',families,index=families.index(ss.settings[family]) if ss.settings[family] in families else 0,key='layout_'+family+'_'+str(ss.project_revision))
        ss.settings[size]=col.number_input(label+' font size / pt',1.,72.,float(ss.settings[size]),.1,key='layout_'+size+'_'+str(ss.project_revision))
        styles=available_styles(ss.settings[family]);style_key=role+'_font_style';current=ss.settings.get(style_key,'regular')
        ss.settings[style_key]=col.selectbox(label+' font style',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k],key='layout_'+style_key+'_'+str(ss.project_revision))
    ss.settings['font_family']=ss.settings['protein_font_family'];ss.settings['font_pt']=ss.settings['protein_font_pt']
    st.caption('Local formatting: *Pfkl*^{S775A/S775A} italicizes only Pfkl; **text** is bold; \\mathrm{text} is regular. Choose Regular for the overall font to keep other text upright.')
    st.caption('Group headings and units use the lane font. Font sizes can be set independently. Installed Arial, Helvetica and Times New Roman are listed automatically.')
    with st.expander('kDa unit: independent font, size and position',expanded=False):
        kcols=st.columns(3)
        ss.settings['kda_font_family']=kcols[0].selectbox('kDa font',families,index=families.index(ss.settings['kda_font_family']) if ss.settings['kda_font_family'] in families else 0)
        ss.settings['kda_font_pt']=kcols[1].number_input('kDa font size / pt',1.,72.,float(ss.settings['kda_font_pt']),.1)
        styles=available_styles(ss.settings['kda_font_family']);current=ss.settings['kda_font_style']
        ss.settings['kda_font_style']=kcols[2].selectbox('kDa font style',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k])
        pos=st.columns(2);kd=ss.settings['kda_layout']
        kd['x_offset_mm']=pos[0].number_input('kDa horizontal offset / mm',-100.,100.,float(kd['x_offset_mm']),.1)
        kd['y_offset_mm']=pos[1].number_input('kDa vertical offset / mm',-100.,100.,float(kd['y_offset_mm']),.1)
        st.caption('Offsets from the default position: positive moves right/down, negative moves left/up. Marker numbers do not move.')
    ss.settings['header_unit']=st.text_input('Header unit (optional)',value=ss.settings['header_unit'])
    with st.expander('Marker positions',expanded=False):
        st.caption('0% is the strip top; 100% is the bottom. These controls move text and ticks, not the original marker bands.')
        marker_protein=st.selectbox('Protein for marker adjustment',[item.id for item in ss.proteins],format_func=lambda key:ref_names[key])
        marker_item=next(item for item in ss.proteins if item.id==marker_protein)
        style=ss.settings['marker_layout'].setdefault(marker_protein,{'position_pct':50.,'x_offset_mm':0.})
        style['position_pct']=st.slider('Marker vertical position / %',0.,100.,float(style['position_pct']),1.,key='marker_pos_'+marker_protein)
        style['x_offset_mm']=st.slider('Marker horizontal offset / mm',-5.,.3,float(style['x_offset_mm']),.1,key='marker_x_'+marker_protein)
        if st.button('Add marker label',key='add_marker_'+marker_protein):
            if not marker_item.markers and marker_item.molecular_weight:
                marker_item.markers=[{'label':marker_item.molecular_weight.replace('kDa','').strip(),'position_pct':style['position_pct'],'x_offset_mm':style['x_offset_mm']}]
            marker_item.markers.append({'label':'','position_pct':50.,'x_offset_mm':style['x_offset_mm']})
        if marker_item.markers:
            st.caption('Multiple marker labels are present. Adjust their positions below.')
            x,y,w,h=marker_item.crop
            entries=[{'label':m['label'],'position_pct':m.get('position_pct',(m.get('y',y+h/2)-y)/h*100),'x_offset_mm':m.get('x_offset_mm',style['x_offset_mm'])} for m in marker_item.markers]
            edited=st.data_editor(pd.DataFrame(entries),hide_index=True,num_rows='dynamic',key='marker_panel_'+marker_protein,column_config={'position_pct':st.column_config.NumberColumn('Vertical position %',min_value=0,max_value=100),'x_offset_mm':st.column_config.NumberColumn('Horizontal offset mm',min_value=-5,max_value=.3),'label':'Marker text'})
            marker_item.markers=[{'label':str(row.label),'position_pct':float(row.position_pct),'x_offset_mm':float(row.x_offset_mm)} for row in edited.itertuples() if pd.notna(row.label) and pd.notna(row.position_pct) and pd.notna(row.x_offset_mm)]
    st.markdown('**Free-text annotations**')
    if st.button('Add text',key='add_figure_text'):
        ss.settings['figure_texts'].append({'id':uuid.uuid4().hex,'text':'','x_pct':50.,'y_pct':10.,'font_pt':ss.settings['protein_font_pt'],'font_family':ss.settings['protein_font_family']})
    if ss.settings['figure_texts']:
        st.caption('Greek letters and super/subscripts are supported. Positions are percentages of the full figure. Multiple text annotations can be added.')
    for index,item in enumerate(list(ss.settings['figure_texts'])):
        tid=item['id']
        with st.expander('Text '+str(index+1)+((' · '+item['text']) if item['text'] else ''),expanded=True):
            item['text']=st.text_input('Text content',value=item['text'],key='free_text_'+tid)
            position=st.columns(2)
            item['x_pct']=position[0].slider('Text horizontal position / %',0.,100.,float(item['x_pct']),.1,key='free_x_'+tid)
            item['y_pct']=position[1].slider('Text vertical position / %',0.,100.,float(item['y_pct']),.1,key='free_y_'+tid)
            style=st.columns(2)
            item['font_family']=style[0].selectbox('Text font',families,index=families.index(item['font_family']) if item['font_family'] in families else 0,key='free_font_'+tid)
            item['font_pt']=style[1].number_input('Text font size / pt',1.,72.,float(item['font_pt']),.1,key='free_size_'+tid)
            styles=available_styles(item['font_family']);current=item.get('font_style','regular')
            item['font_style']=st.selectbox('Text font style',styles,index=styles.index(current) if current in styles else 0,format_func=lambda k:FONT_STYLES[k],key='free_style_'+tid)
            if st.button('Delete this text',key='free_delete_'+tid):
                ss.settings['figure_texts']=[other for other in ss.settings['figure_texts'] if other['id']!=tid]
                invalidate();st.rerun()
    with st.expander('More layout settings',expanded=False):
        ss.settings['label_margin_mm']=st.number_input('Protein-label margin / mm',10.,60.,float(ss.settings['label_margin_mm']),1.)
        ss.settings['block_lanes']=st.number_input('Lanes per block (0 = no blocks)',0,len(ss.samples),min(int(ss.settings['block_lanes']),len(ss.samples)))
        ss.settings['block_gap_mm']=st.number_input('Block spacing / mm',.2,5.,float(ss.settings['block_gap_mm']),.2)
    st.caption('The first confirmed crop defines strip dimensions. Strips are stacked without per-lane resizing; 600 DPI applies the same scale factor to all strips.')
    try:
        options={k:ss.settings[k] for k in ['dpi','font_pt','gap_mm','label_margin_mm','block_lanes','block_gap_mm','header_unit','font_family','marker_font_pt','marker_layout','lane_font_pt','protein_font_pt','lane_font_family','protein_font_family','marker_font_family','figure_texts','lane_font_style','protein_font_style','marker_font_style','kda_font_pt','kda_font_family','kda_font_style','kda_layout']}
        panel,panel_meta=publication_figure(ss.sources,ss.proteins,ss.samples,**options,native_size=True)
        st.image(panel,caption=f"{panel.width} × {panel.height} px · {ss.settings['dpi']} DPI · Separate protein strips; borders and gaps indicate assembly",width='stretch')
        if any(r['upsampled'] for r in panel_meta['strips']):st.warning('Some strips are upsampled. DPI and interpolation do not add original detail; check journal requirements.')
        st.caption('The figure TIFF is a lossless LZW-compressed 8-bit RGB display image. Original 16-bit files are preserved in the project ZIP. PDF and TIFF share the same physical width.')
        checkpoint()
        if st.button('Generate full export bundle',type='primary'):
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
        st.error('Cannot generate figure: '+str(error))
    st.caption('300/600 DPI are output settings, not a guarantee of journal compliance. Check figure dimensions, formats, source-data and splice-disclosure requirements.')

with tabs[5]:
    checkpoint()
    st.subheader('Save an editable project')
    st.caption('Save source images, crops, annotations and quantification settings. Upload the project ZIP next time to continue.')
    data=project_bytes(ss.sources,ss.proteins,ss.samples,ss.audit,ss.settings)
    st.download_button('Save project ZIP (includes originals)',data,file_name='wb-project.zip',mime='application/zip')
    with st.expander('Processing history (optional)',expanded=False):
        st.caption('Source provenance, crop coordinates, rotation, inversion and quantification parameters are recorded for review; this does not certify experimental authenticity.')
        if not ss.audit.verify():st.warning('Legacy processing history is incomplete; this alone does not indicate a problem with source images.')
        st.json(ss.audit.document(ss.settings,[src.metadata() for src in ss.sources.values()]),expanded=False)
