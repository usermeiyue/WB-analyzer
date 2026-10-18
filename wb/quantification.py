"""Single editable ROI state and explicit-button native-pixel quantification."""
import copy
import json
import numpy as np
import pandas as pd
from .core import measure, rect_slice, sha, canonical
from .imaging import crop_corners


def empty_workspace(protein, lanes):
    return {'source_id':protein.source_id,'lanes':[{'signal':None,'background':None,'origin':'manual','detected':None,'no_band':False} for _ in range(lanes)],'revision':0}


def search_rect(source, protein):
    if not protein.crop_selected:return [0,0,source.gray.shape[1],source.gray.shape[0]]
    points=crop_corners(protein.crop,protein.crop_angle)
    x=max(0,int(np.floor(points[:,0].min())));y=max(0,int(np.floor(points[:,1].min())))
    right=min(source.gray.shape[1],int(np.ceil(points[:,0].max())));bottom=min(source.gray.shape[0],int(np.ceil(points[:,1].max())))
    return [x,y,right-x,bottom-y]


def propose_workspace(source, protein, lanes, threshold=5., fraction=.8, region=None):
    """Locate each lane's row against a robust low-signal baseline, then suggest a
    disjoint background rectangle on the full original (even for a tight crop).
    Automatic no-band flags are explicit editable decisions, not a detection limit.
    """
    if source.ceiling<=source.floor:raise ValueError('Sensor maximum must exceed minimum.')
    if lanes<1 or not 2<=threshold<=15 or not .4<=fraction<=1:raise ValueError('Invalid detection parameters.')
    region=region or search_rect(source,protein);rect_slice(region,source.gray.shape)
    x,y,w,h=region
    if h<4 or w/lanes<3:raise ValueError('Search region is too small. Enlarge it or reduce the lane count.')
    signal=source.ceiling-source.gray.astype(float) if protein.polarity=='dark' else source.gray.astype(float)-source.floor
    pitch=w/lanes;proposals=[];peaks=[]
    for lane in range(lanes):
        left=max(0,round(x+(lane+.5)*pitch-pitch*fraction/2));right=min(signal.shape[1],round(x+(lane+.5)*pitch+pitch*fraction/2))
        block=signal[y:y+h,left:right]
        base=float(np.percentile(block,20))
        low=block[block<=np.percentile(block,50)]
        # A tight figure crop may contain band everywhere vertically. Estimate detection
        # background on adjacent native rows instead of calling that band 'background'.
        margin=max(4,min(20,h//2));outside=[]
        for by in [y-margin-3,y-3*margin-5,y+h+3,y+h+2*margin+5]:
            if 0<=by and by+margin<=signal.shape[0]:outside.append(signal[by:by+margin,left:right])
        if outside:
            adjacent=min(outside,key=lambda a:float(np.median(a)))
            if float(np.median(adjacent))<base:
                low=adjacent.ravel();base=float(np.median(adjacent))
        noise=max(1.,float(1.4826*np.median(np.abs(low-np.median(low)))))
        profile=block.mean(axis=1)-base
        peak=int(np.argmax(profile));snr=float(profile[peak]/noise)
        lo=hi=peak;cutoff=max(noise*2,float(profile[peak])*.12)
        while lo>0 and profile[lo-1]>cutoff:lo-=1
        while hi<h-1 and profile[hi+1]>cutoff:hi+=1
        detected=snr>=threshold and hi-lo+1>=2
        if detected:peaks.append((y+peak,hi-lo+1))
        proposals.append((left,right,y+lo,y+hi+1,detected,snr))
    common_y=int(np.median([r[0] for r in peaks])) if peaks else y+h//2
    common_h=max(4,int(np.median([r[1] for r in peaks]))) if peaks else min(h,max(4,h//4))
    workspace=empty_workspace(protein,lanes);workspace['search_region']=region
    for left,right,top,bottom,detected,snr in proposals:
        if not detected:top=common_y-common_h//2;bottom=top+common_h
        top=max(0,top-2);bottom=min(signal.shape[0],bottom+2)
        rect=[left,top,max(1,right-left),max(1,bottom-top)]
        bh=max(2,min(12,max(2,rect[3]//3)));candidates=[]
        # Look nearby, rather than fitting a baseline through crop edges containing a broad band.
        for by in [top-bh-3,top-2*bh-5,bottom+3,bottom+bh+5]:
            if 0<=by and by+bh<=signal.shape[0]:
                candidates.append((float(signal[by:by+bh,left:right].mean()),[left,by,right-left,bh]))
        background=min(candidates,key=lambda r:r[0])[1] if candidates else None
        workspace.setdefault('_made',[]).append({'signal':rect,'background':background,'origin':'automatic','detected':bool(detected),'no_band':False,'detection_snr':snr,'threshold':threshold})
    workspace['lanes']=workspace.pop('_made');workspace['revision']=1
    return workspace


def quant_fingerprint(sources, proteins, samples, reference_id, workspaces, exclude_saturated):
    return sha(canonical({'version':'editable-rois-v3-background','reference':reference_id,'samples':samples,'exclude_saturated':exclude_saturated,
        'proteins':[{'id':p.id,'name':p.name,'source':sha(sources[p.source_id].original),'floor':sources[p.source_id].floor,'ceiling':sources[p.source_id].ceiling,
        'polarity':p.polarity,'membrane':p.membrane_id,'background_method':p.background_method,'workspace':workspaces.get(p.id)} for p in proteins]}))


def calculate(sources, targets, reference, samples, workspaces, exclude_saturated=True):
    if not targets:raise ValueError('Select at least one target to measure.')
    if not samples:raise ValueError('Enter lane and sample information.')
    if reference and any(p.id==reference.id for p in targets):raise ValueError('A loading control cannot also be a target.')
    measured=[]
    for p in targets+([reference] if reference else []):
        source=sources[p.source_id];workspace=workspaces.get(p.id)
        if not workspace or workspace.get('source_id')!=p.source_id or len(workspace['lanes'])!=len(samples):
            raise ValueError('Generate measurement regions for '+p.name+', or draw band and background regions manually.')
        for lane,roi in enumerate(workspace['lanes']):
            background=None
            try:
                background=background_for_roi(source,p,workspace,roi)
                if roi.get('signal') is None or background is None:raise ValueError('missing_signal_or_background')
                temporary=copy.copy(p);temporary.signal=[roi['signal']];temporary.background=[background]
                m=measure(source,temporary,0,exclude_saturated,True)
            except (ValueError,IndexError,TypeError) as error:
                m={'valid':False,'status':str(error),'band_intensity':np.nan,'raw_integrated':np.nan,'background_integrated':np.nan,'area_px':0}
            net=m['band_intensity'];automatic=roi.get('origin')=='automatic' and roi.get('detected') is False
            user_zero=bool(roi.get('no_band'));zero=user_zero or automatic or (np.isfinite(net) and net<=0)
            value=(0. if zero else net) if m['valid'] else np.nan
            reason='user_marked_no_visible_band' if user_zero else ('automatic_below_detection_threshold' if automatic else ('not_above_background' if zero else ''))
            note='Manually marked absent; assigned zero' if user_zero else ('Automatically undetected; assigned zero' if automatic else ('Signal not above background; assigned zero' if zero else 'Measured'))
            measured.append({**samples[lane],**m,'protein_id':p.id,'protein':p.name,'lane':lane+1,'role':'reference' if reference and p.id==reference.id else 'target',
                'band_intensity':value,'analysis_intensity':value,'measured_net_intensity':net,'analysis_status':m['status'],'measurement_note':note,
                'signal_xywh':json.dumps(roi.get('signal')),'background_xywh':json.dumps(background),
                'roi_origin':roi.get('origin'),'background_origin':roi.get('background_origin','automatic' if workspace.get('background_mode')=='automatic_adjacent' else 'manual'),'band_detected':not zero,'zero_override_applied':bool(user_zero and m['valid']),
                'zero_assignment_source':reason,'source_id':source.id,'source_sha256':sha(source.original),'source_page':source.page,
                'detection_snr':roi.get('detection_snr',np.nan) if roi.get('origin')=='automatic' else np.nan,
                'reference':reference.name if reference else '','membrane_id':p.membrane_id,'analysis_method':'current band rectangle / native pixels / '+('automatic adjacent background' if workspace.get('background_mode')=='automatic_adjacent' and roi.get('background_origin')!='manual' else 'manual background')})
    details=pd.DataFrame(measured);refrows=details[details.role=='reference'].to_dict('records');records=[]
    for row in details[details.role=='target'].to_dict('records'):
        ratio=np.nan;issues=[]
        if not row['valid']:issues.append(row['analysis_status'])
        ref=refrows[row['lane']-1] if reference else None
        if ref:
            p=next(p for p in targets if p.id==row['protein_id'])
            if p.membrane_id and reference.membrane_id and p.membrane_id!=reference.membrane_id:issues.append('membrane_pairing_mismatch')
            if not ref['valid'] or not np.isfinite(ref['analysis_intensity']) or ref['analysis_intensity']<=0:issues.append('invalid_or_nonpositive_reference')
            if not issues:ratio=row['analysis_intensity']/ref['analysis_intensity']
        records.append({**row,'reference_intensity':ref['analysis_intensity'] if ref else np.nan,'reference_status':ref['analysis_status'] if ref else 'not_selected',
                        'reference_measurement_note':ref['measurement_note'] if ref else '', 'reference_signal_xywh':ref['signal_xywh'] if ref else '',
                        'normalized_ratio':ratio,'analysis_value':ratio if reference else row['analysis_intensity'],'analysis_status':';'.join(issues) or 'ok'})
    raw=pd.DataFrame(records)
    keys=[(s.get('condition',''),s.get('bio_rep',''),s.get('tech_rep','')) for s in samples]
    stats=all(all(str(v).strip() for v in k) for k in keys) and len(set(keys))==len(keys)
    bio=raw.groupby(['protein_id','protein','condition','bio_rep'],dropna=False,sort=False).analysis_value.agg(technical_n='count',bio_value='mean').reset_index() if stats else pd.DataFrame(columns=['protein_id','protein','condition','bio_rep','technical_n','bio_value'])
    summary=bio.groupby(['protein_id','protein','condition'],dropna=False,sort=False).agg(biological_n=('bio_value','count'),mean=('bio_value','mean'),sd=('bio_value','std')).reset_index();summary['reference']=reference.name if reference else ''
    return raw,bio,summary,details


def suggest_background(source, protein, rect):
    rect_slice(rect,source.gray.shape)
    x,y,w,h=rect;bh=max(2,min(16,max(2,h//3)));candidates=[]
    for by in [y-bh-3,y-2*bh-5,y+h+3,y+h+bh+5]:
        if 0<=by and by+bh<=source.gray.shape[0]:
            block=source.gray[by:by+bh,x:x+w].astype(float)
            signal=source.ceiling-block if protein.polarity=='dark' else block-source.floor
            candidates.append((float(signal.mean()),[x,by,w,bh]))
    return min(candidates,key=lambda item:item[0])[1] if candidates else None


def workspace_from_row(source, protein, lanes, band_region, background_region=None, threshold=5., fraction=.8):
    """One row selection creates every lane ROI; explicit row background is paired
    by lane order. Native measurement rectangles are equal height and width.
    """
    rect_slice(band_region,source.gray.shape)
    if background_region is not None:rect_slice(background_region,source.gray.shape)
    result=propose_workspace(source,protein,lanes,threshold,fraction,region=band_region)
    x,y,w,h=band_region
    for i,roi in enumerate(result['lanes']):
        left=round(x+(i+.5)*w/lanes-w/lanes*fraction/2)
        width=max(1,round(w/lanes*fraction))
        roi['signal']=[left,y,width,h]
        rect_slice(roi['signal'],source.gray.shape)
        if roi.get('background_origin')!='manual':roi['background']=suggest_background(source,protein,roi['signal'])
        if background_region is not None:
            bx,by,bw,bh=background_region
            bl=round(bx+(i+.5)*bw/lanes-bw/lanes*fraction/2)
            roi['background']=[bl,by,max(1,round(bw/lanes*fraction)),bh]
            rect_slice(roi['background'],source.gray.shape)
    result['row_dirty']=False
    result['row_regions']={'signal':list(band_region),'background':list(background_region) if background_region is not None else None}
    return result


def match_first_lane_size(source, protein, workspace):
    """Keep existing lane centers, clamp to native bounds; never restore deleted ROIs."""
    first=workspace['lanes'][0].get('signal')
    if first is None:return
    h,w=source.gray.shape;fw,fh=first[2:]
    if fw>w or fh>h:raise ValueError('Lane 1 region exceeds the source image.')
    for roi in workspace['lanes'][1:]:
        old=roi.get('signal')
        if old is None:continue
        if old[2:]==[fw,fh]:continue
        x=max(0,min(w-fw,round(old[0]+old[2]/2-fw/2)))
        y=max(0,min(h-fh,round(old[1]+old[3]/2-fh/2)))
        roi['signal']=[x,y,fw,fh]
        if roi.get('background_origin')!='manual':roi['background']=suggest_background(source,protein,roi['signal'])


def background_for_roi(source, protein, workspace, roi):
    """One source of truth for the displayed and measured background rectangle."""
    if roi.get('background_origin')=='manual':return roi.get('background')
    if workspace.get('background_mode')=='automatic_adjacent' and roi.get('signal') is not None:
        return suggest_background(source,protein,roi['signal'])
    return roi.get('background')
