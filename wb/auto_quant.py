"""Per-lane automatic detection and integration on untouched native pixels.
Detection zeros are threshold decisions, not measured proof of biological absence.
"""
import json
import math
import copy
import numpy as np
import pandas as pd
from .core import sha, measure, canonical
from .imaging import validate_crop,crop_corners


def crop_signature(protein):
    return sha(canonical({'source':protein.source_id,'crop':protein.crop,'angle':protein.crop_angle,'polarity':protein.polarity}))


def auto_measure(source,protein,lanes,*,threshold_sigma=5.,lane_fraction=.8,exclude_saturated=True,allow_demo=False,roi_overrides=None):
    if source.ceiling<=source.floor:raise ValueError('Sensor maximum must exceed minimum.')
    if not protein.crop_selected and not allow_demo:raise ValueError('Confirm the crop for '+protein.name+' first.')
    if not 2<=threshold_sigma<=15 or not .4<=lane_fraction<=1:raise ValueError('Automatic detection parameters are out of range.')
    validate_crop(protein.crop,protein.crop_angle,source.gray.shape)
    if protein.segments:raise ValueError('Automatic quantification requires a continuous, unspliced source region.')
    x,y,w,h=protein.crop
    if h<8 or w/lanes<4:raise ValueError('The strip must be at least 8 pixels high, with at least 4 pixels per lane.')
    corners=crop_corners(protein.crop,protein.crop_angle)
    x0=max(0,int(np.floor(corners[:,0].min())));x1=min(source.gray.shape[1],int(np.ceil(corners[:,0].max())))
    y0=max(0,int(np.floor(corners[:,1].min())));y1=min(source.gray.shape[0],int(np.ceil(corners[:,1].max())))
    yy,xx=np.mgrid[y0:y1,x0:x1];c=math.cos(math.radians(protein.crop_angle));s=math.sin(math.radians(protein.crop_angle))
    u=(xx+.5-x-w/2)*c+(yy+.5-y-h/2)*s+w/2
    v=-(xx+.5-x-w/2)*s+(yy+.5-y-h/2)*c+h/2
    gray=source.gray[y0:y1,x0:x1].astype(float)
    signal=source.ceiling-gray if protein.polarity=='dark' else gray-source.floor
    bins=np.floor(v).astype(int);inside=(v>=0)&(v<h)
    edge=max(2,round(h*.18));pitch=w/lanes;results=[]
    for lane in range(lanes):
        center=(lane+.5)*pitch
        mask=inside&(np.abs(u-center)<pitch*lane_fraction/2)
        bgmask=mask&((v<edge)|(v>=h-edge))
        bgvalues=signal[bgmask]
        if bgvalues.size<4:raise ValueError('Insufficient background pixels. Include more background above and below the strip.')
        # Fit a linear baseline to the upper/lower edge medians, tolerating smooth gradients.
        upper=mask&(v<edge);lower=mask&(v>=h-edge)
        a=float(np.median(signal[upper]));b=float(np.median(signal[lower]))
        va=float(np.median(v[upper]));vb=float(np.median(v[lower]))
        baseline=a+(b-a)*(v-va)/max(1,vb-va)
        residual=signal-baseline
        bgres=residual[bgmask];noise=max(1.,float(1.4826*np.median(np.abs(bgres-np.median(bgres)))))
        counts=np.bincount(bins[mask],minlength=h);sums=np.bincount(bins[mask],weights=residual[mask],minlength=h)
        profile=np.divide(sums,counts,out=np.zeros(h),where=counts>0)
        interior=profile.copy();interior[:edge]=0;interior[h-edge:]=0
        peak=int(np.argmax(interior));snr=float(interior[peak]/noise)
        lo=hi=peak
        cutoff=max(2*noise,interior[peak]*.12)
        while lo>edge and profile[lo-1]>cutoff:lo-=1
        while hi<h-edge-1 and profile[hi+1]>cutoff:hi+=1
        detected=bool(snr>=threshold_sigma and hi-lo+1>=2)
        # Integrate a common central lane width over the detected vertical band extent.
        band=mask&(bins>=lo)&(bins<=hi) if detected else mask&(bins>=edge)&(bins<h-edge)
        raw=float(signal[band].sum());background=float(baseline[band].sum());net=raw-background
        detected=detected and net>0
        data=source.native_rgb[y0:y1,x0:x1] if source.native_rgb is not None else source.gray[y0:y1,x0:x1]
        sat=(data<=source.floor) if protein.polarity=='dark' else (data>=source.ceiling)
        if sat.ndim==3:sat=sat.any(axis=-1)
        saturation_fraction=float(sat[band].mean())
        out_of_range=bool(((data[band]<source.floor)|(data[band]>source.ceiling)).any() or ((data[bgmask]<source.floor)|(data[bgmask]>source.ceiling)).any())
        valid=not (exclude_saturated and (sat[band].any() or sat[bgmask].any())) and not out_of_range
        manual=lane in protein.not_detected_lanes
        assigned=not detected or manual
        intensity=0. if assigned else net
        positions=np.column_stack((xx[band],yy[band]));bounds=[int(positions[:,0].min()),int(positions[:,1].min()),int(np.ptp(positions[:,0]))+1,int(np.ptp(positions[:,1]))+1]
        results.append({'lane':lane+1,'band_intensity':intensity if valid else np.nan,'analysis_intensity':intensity if valid else np.nan,
                        'raw_integrated':raw,'background_integrated':background,'measured_net_intensity':net,
                        'background_noise_sigma':noise,'lane_fraction':lane_fraction,'band_detected':detected,'detection_snr':snr,'detection_threshold_sigma':threshold_sigma,
                        'not_detected_manual':manual,'zero_override_applied':bool(manual and valid),
                        'zero_assignment_source':'user_marked_no_visible_band' if manual else ('automatic_below_detection_threshold' if not detected else ''),
                        'signal_xywh':json.dumps(bounds),'auto_band_rows':json.dumps([lo,hi]),'area_px':int(band.sum()),
                        'saturated_fraction':saturation_fraction,'valid':valid,'measurement_note':'No band detected; assigned zero' if assigned else 'Band detected automatically',
                        'analysis_status':'ok' if valid else ('pixels_outside_sensor_range' if out_of_range else 'saturated_signal_or_background'),
                        'background_subtracted':True,'polarity':protein.polarity})
    saved=(roi_overrides or {}).get(protein.id,{})
    if saved.get('signature')==crop_signature(protein):
        for key,roi in saved.get('lanes',{}).items():
            lane=int(key)-1
            if not 0<=lane<lanes:continue
            item=copy.copy(protein);item.signal=[roi['signal']];item.background=[roi['background']]
            m=measure(source,item,0,exclude_saturated,True)
            row=results[lane];net=m['band_intensity'];manual=lane in protein.not_detected_lanes
            detected=bool(net>0);value=0. if manual or not detected else net
            row.update(m,band_intensity=value if m['valid'] else np.nan,analysis_intensity=value if m['valid'] else np.nan,
                       measured_net_intensity=net,band_detected=detected,signal_xywh=json.dumps(roi['signal']),
                       background_xywh=json.dumps(roi['background']),detection_snr=np.nan,
                       zero_override_applied=bool(manual and m['valid']),
                       zero_assignment_source='user_marked_no_visible_band' if manual else ('manual_roi_not_above_background' if not detected else ''),
                       measurement_note='User-confirmed region: assigned zero' if manual or not detected else 'User-confirmed region: band measured',
                       analysis_status=m['status'],roi_method='user_confirmed_rectangle')
    return results


def auto_quantify(sources,targets,reference,samples,**options):
    if reference is None or not targets:raise ValueError('Select a loading control and at least one target.')
    if any(p.id==reference.id for p in targets):raise ValueError('A target cannot also be its loading control.')
    refrows=auto_measure(sources[reference.source_id],reference,len(samples),**options);records=[]
    for p in targets:
        if p.membrane_id and reference.membrane_id and p.membrane_id!=reference.membrane_id:raise ValueError('Target and control membrane IDs differ. Verify pairing.')
        for lane,m in enumerate(auto_measure(sources[p.source_id],p,len(samples),**options)):
            r=refrows[lane];ratio=np.nan;status=m['analysis_status']
            if not r['valid'] or not np.isfinite(r['analysis_intensity']) or r['analysis_intensity']<=0:status+=';invalid_or_nonpositive_reference'
            elif m['valid']:ratio=m['analysis_intensity']/r['analysis_intensity']
            records.append({**samples[lane],**m,'protein_id':p.id,'protein':p.name,'reference':reference.name,
                            'reference_intensity':r['analysis_intensity'],'reference_band_detected':r['band_detected'],
                            'reference_signal_xywh':r['signal_xywh'],'reference_status':r['analysis_status'],'reference_measurement_note':r['measurement_note'],'normalized_ratio':ratio,'analysis_value':ratio,
                            'analysis_status':status,'source_id':p.source_id,'source_sha256':sha(sources[p.source_id].original),
                            'source_page':sources[p.source_id].page,'membrane_id':p.membrane_id,'pairing_method':'user-selected corresponding lane order',
                            'analysis_method':'user-confirmed rectangles / local background' if m.get('roi_method') else 'native pixel masks / edge baseline / automatic band detection'})
    raw=pd.DataFrame(records)
    keys=[(s.get('condition',''),s.get('bio_rep',''),s.get('tech_rep','')) for s in samples]
    stats=all(all(str(v).strip() for v in k) for k in keys) and len(set(keys))==len(keys)
    bio=raw.groupby(['protein_id','protein','condition','bio_rep'],dropna=False,sort=False).analysis_value.agg(technical_n='count',bio_value='mean').reset_index() if stats else pd.DataFrame(columns=['protein_id','protein','condition','bio_rep','technical_n','bio_value'])
    summary=bio.groupby(['protein_id','protein','condition'],dropna=False,sort=False).agg(biological_n=('bio_value','count'),mean=('bio_value','mean'),sd=('bio_value','std')).reset_index()
    summary['reference']=reference.name
    return raw,bio,summary,refrows
