"""Band-row suggestions and geometric display crops; never modify raw measurements."""
from __future__ import annotations
import math
import cv2
import numpy as np
from PIL import Image
from .core import Source, Protein, display_image, rect_slice


def crop_corners(rect, angle=0):
    x,y,w,h=rect;c=math.cos(math.radians(angle));s=math.sin(math.radians(angle))
    cx=x+w/2;cy=y+h/2
    return np.array([[cx+u*c-v*s,cy+u*s+v*c] for u,v in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])


def validate_crop(rect, angle, shape):
    rect_slice(rect,shape)
    if not math.isfinite(angle) or abs(angle)>45:raise ValueError('旋转角度须在 −45° 到 45° 之间。')
    corners=crop_corners(rect,angle)
    if (corners[:,0]<-.01).any() or (corners[:,1]<-.01).any() or (corners[:,0]>shape[1]+.01).any() or (corners[:,1]>shape[0]+.01).any():
        raise ValueError('旋转后的裁剪框超出原图，请缩小或向图内移动。')


def fit_aspect(rect, aspect, shape):
    x,y,w,h=rect
    if aspect<=0:raise ValueError('裁片宽高比必须大于零。')
    hh=max(1,round(w/aspect));hh=min(hh,shape[0])
    ww=min(w,max(1,round(hh*aspect)),shape[1])
    return [max(0,min(shape[1]-ww,round(x+(w-ww)/2))),max(0,min(shape[0]-hh,round(y+(h-hh)/2))),ww,hh]


def extract_strip(source: Source, protein: Protein):
    """Map output pixels to a rotated rectangle in the original display image."""
    angle=protein.crop_angle
    validate_crop(protein.crop,angle,source.gray.shape)
    x,y,w,h=protein.crop
    if not angle:
        return display_image(source).crop((x,y,x+w,y+h))
    if protein.segments:raise ValueError('旋转裁剪与非连续区段不能同时使用；请分别添加蛋白裁片。')
    c=math.cos(math.radians(angle));s=math.sin(math.radians(angle));cx=x+w/2;cy=y+h/2
    # Pixel-center convention makes angle=0 identical to native cropping.
    matrix=np.array([[c,-s,cx-c*(w-1)/2+s*(h-1)/2-.5],
                     [s,c,cy-s*(w-1)/2-c*(h-1)/2-.5]],np.float64)
    arr=cv2.warpAffine(np.asarray(display_image(source)),matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(arr)


def detect_band_rows(source: Source, polarity='auto', max_rows=10, padding=3):
    """Suggest horizontal band rows with morphological background rejection + deskew.

    Results locate rows, not protein identity or molecular weight. Scores are signal
    strengths, not probabilities. No detections are applied until user confirmation.
    """
    raw=source.gray.astype(np.float32)
    lo,hi=np.percentile(raw,[1,99.5])
    if hi-lo<1e-6:return []
    scaled=np.clip((raw-lo)/(hi-lo),0,1)
    factor=min(1.,900/max(raw.shape));a=cv2.resize(scaled,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    h,w=a.shape
    if min(h,w)<24:return []
    size=max(15,min(81,int(h*.12)//2*2+1));kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    best=None
    for pol in (['dark','light'] if polarity=='auto' else [polarity]):
        signal=1-a if pol=='dark' else a
        response=cv2.morphologyEx(signal,cv2.MORPH_TOPHAT,kernel)
        # Inspect central 90% to reduce marker/film-edge influence.
        left=max(1,round(w*.05));right=w-left
        for angle in np.arange(-8,8.01,1):
            m=cv2.getRotationMatrix2D((w/2,h/2),float(angle),1)
            r=cv2.warpAffine(response,m,(w,h),borderMode=cv2.BORDER_CONSTANT)
            central=r[:,left:right]
            coverage=(central>max(.025,float(np.percentile(response,85)))).mean(axis=1)
            profile=central.mean(axis=1)*np.clip(coverage/.3,0,1)
            profile=cv2.GaussianBlur(profile.reshape(-1,1),(1,5),0).ravel()
            edge=max(5,size//2);profile[:edge]=0;profile[-edge:]=0
            quality=float(np.max(profile))
            if best is None or quality>best[0]:best=(quality,profile,float(angle),pol,m,r)
    strength,profile,angle,pol,m,response=best
    if strength<.012:return []
    threshold=max(.012,float(np.median(profile)+3*np.median(np.abs(profile-np.median(profile)))),strength*.18)
    peaks=[i for i in range(1,h-1) if profile[i]>=threshold and profile[i]>profile[i-1] and profile[i]>=profile[i+1]]
    chosen=[]
    for y in sorted(peaks,key=lambda i:profile[i],reverse=True):
        if all(abs(y-old)>max(12,size*.45) for old in chosen):chosen.append(y)
        if len(chosen)>=max_rows:break
    inverse=cv2.invertAffineTransform(m);results=[]
    for y in sorted(chosen):
        lower=upper=y
        while lower>0 and profile[lower]>profile[y]*.25:lower-=1
        while upper<h-1 and profile[upper]>profile[y]*.25:upper+=1
        # Locate actual band components in this row, not a fixed fraction of film width.
        region=response[lower:upper+1]
        cutoff=max(.025,float(region.max())*.10)
        binary=(region>cutoff).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        parts=[stats[i] for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=max(4,(upper-lower)*.4)
               and stats[i,cv2.CC_STAT_WIDTH]>=2]
        if not parts:continue
        x0=min(int(part[0]) for part in parts);x1=max(int(part[0]+part[2]) for part in parts)
        y0=lower+min(int(part[1]) for part in parts);y1=lower+max(int(part[1]+part[3]) for part in parts)
        pad=max(0,float(padding))*factor
        cx,cy=inverse@np.array([(x0+x1)/2,(y0+y1)/2,1.])
        cw=max(1,round((x1-x0+2*pad)/factor));ch=max(1,round((y1-y0+2*pad)/factor))
        rect=[round(cx/factor-cw/2),round(cy/factor-ch/2),cw,ch]
        try:validate_crop(rect,angle,raw.shape)
        except ValueError:continue
        corners=crop_corners(rect,angle)
        ax=max(0,int(np.floor(corners[:,0].min())));ay=max(0,int(np.floor(corners[:,1].min())))
        bx=min(raw.shape[1],int(np.ceil(corners[:,0].max())));by=min(raw.shape[0],int(np.ceil(corners[:,1].max())))
        results.append({'rect':rect,'axis_rect':[ax,ay,bx-ax,by-ay],'angle':angle,'polarity':pol,
                        'score':round(float(profile[y]),4),'source_y':round(cy/factor),'padding':padding})
    return results


def fixed_crop(rect, size, shape, angle=0.):
    """Place a fixed-size rectangle near a suggestion without ever resizing it."""
    w,h=map(int,size);image_h,image_w=shape
    t=math.radians(angle);ex=max(w/2,(abs(w*math.cos(t))+abs(h*math.sin(t)))/2)
    ey=max(h/2,(abs(w*math.sin(t))+abs(h*math.cos(t)))/2)
    xmin=math.ceil(ex-w/2);xmax=math.floor(image_w-ex-w/2)
    ymin=math.ceil(ey-h/2);ymax=math.floor(image_h-ey-h/2)
    if xmin>xmax or ymin>ymax:raise ValueError('这张原图放不下第一张确定的裁剪尺寸；请使用更大原图或减小第一张裁剪框。')
    x=min(xmax,max(xmin,round(rect[0]+rect[2]/2-w/2)))
    y=min(ymax,max(ymin,round(rect[1]+rect[3]/2-h/2)))
    result=[x,y,w,h];validate_crop(result,angle,shape);return result


def quantification_boxes_from_crop(protein, shape, lanes):
    """Uniform proposed ROIs at the chosen strip; requires visual review, never confirmation."""
    if not protein.crop_selected:raise ValueError('请先确认本蛋白裁剪，再生成定量框。')
    x,y,w,h=protein.crop;pitch=w/lanes
    rw=max(1,round(pitch*.65));rh=max(1,round(h*.6));bh=max(1,round(h*.2))
    angle=math.radians(protein.crop_angle);cx=x+w/2;cy=y+h/2
    signals=[];backgrounds=[]
    for i in range(lanes):
        offset=(i+.5)*pitch-w/2
        xx=round(cx+offset*math.cos(angle)-rw/2);yy=round(cy+offset*math.sin(angle)-rh/2)
        rect=[xx,yy,rw,rh];rect_slice(rect,shape)
        by=yy+rh+max(2,round(h*.15))
        if by+bh>shape[0]:by=yy-bh-max(2,round(h*.15))
        bg=[xx,by,rw,bh];rect_slice(bg,shape)
        signals.append(rect);backgrounds.append(bg)
    return signals,backgrounds
