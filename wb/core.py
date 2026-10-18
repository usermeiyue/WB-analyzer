"""Raw-pixel analysis, provenance and reproducible scientific figure rendering."""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import json
import math
import platform
import sys
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.utils import ImageReader

VERSION = "1.0.0"
MAX_PIXELS = 40_000_000


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Audit:
    session: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: list[dict] = field(default_factory=list)

    def add(self, action: str, parameters: dict):
        event = {"sequence": len(self.events), "utc": datetime.now(timezone.utc).isoformat(),
                 "action": action, "parameters": copy.deepcopy(parameters),
                 "previous_sha256": self.events[-1]["sha256"] if self.events else "0" * 64}
        event["sha256"] = sha(canonical(event))
        self.events.append(event)

    def verify(self) -> bool:
        previous = "0" * 64
        for i, event in enumerate(self.events):
            value = dict(event)
            digest = value.pop("sha256", None)
            if value.get("sequence") != i or value.get("previous_sha256") != previous or sha(canonical(value)) != digest:
                return False
            previous = digest
        return True

    def document(self, settings: dict, source_metadata: list[dict]) -> dict:
        return {"application": "WB Image Processor & Analyzer", "version": VERSION,
                "session": self.session, "python": platform.python_version(),
                "chain_valid": self.verify(), "events": self.events, "settings": settings,
                "sources": source_metadata,
                "scope": "Records actions inside this application. Hash chains are not signatures or proof of experimental integrity; retain original acquisition files and externally anchor the final hash."}


@dataclass
class Source:
    id: str
    name: str
    original: bytes
    page: int
    gray: np.ndarray
    ceiling: float
    floor: float = 0.0
    display_black: float = 0.0
    display_white: float = 255.0
    brightness: float = 0.0
    contrast: float = 1.0
    display_invert: bool = False
    native_inverted: bool = False
    palette_white: float = 1.0
    palette_black: float = 0.0
    conversion: str = "native grayscale"
    native_rgb: np.ndarray | None = field(default=None, repr=False)

    def metadata(self):
        return {"id": self.id, "name": self.name, "original_sha256": sha(self.original), "page": self.page,
                "shape": list(self.gray.shape), "dtype": str(self.gray.dtype), "gray_sha256": sha(self.gray.tobytes()),
                "conversion": self.conversion, "native_inverted":self.native_inverted, "palette_white":self.palette_white,"palette_black":self.palette_black, "sensor_floor": self.floor, "sensor_ceiling": self.ceiling,
                "display_black": self.display_black, "display_white": self.display_white,
                "brightness": self.brightness, "contrast": self.contrast, "display_invert": self.display_invert}


def load_sources(data: bytes, name: str) -> list[Source]:
    """Decode each TIFF page explicitly, without silently selecting a z/channel plane."""
    if len(data) > 100 * 1024 * 1024:
        raise ValueError('File exceeds 100 MB. Split the original TIFF pages.')
    frames = [];encodings=[]
    if name.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            for page in tf.pages:
                if math.prod(page.shape) > MAX_PIXELS * 4:
                    raise ValueError('TIFF page exceeds size limits.')
                try:
                    frames.append(page.asarray())
                    inv=page.photometric==tifffile.PHOTOMETRIC.MINISWHITE;white=1.;black=0.;label='native grayscale'
                    if page.photometric==tifffile.PHOTOMETRIC.PALETTE:
                        lut=page.colormap.astype(np.int64)
                        differences=np.diff(lut[0])
                        if not (np.array_equal(lut[0],lut[1]) and np.array_equal(lut[0],lut[2]) and len(differences) and differences[0]!=0 and np.all(differences==differences[0])):
                            raise ValueError('Nonlinear or color TIFF palettes cannot be quantified directly. Export the original grayscale signal page.')
                        inv=bool(differences[0]<0);white=float(lut.max()/65535);black=float(lut.min()/65535)
                        label='native palette indices; affine grayscale palette applied to display only'
                    elif inv:label='native grayscale; TIFF MINISWHITE applied to display only'
                    encodings.append((inv,white,black,label))
                except (ValueError,ModuleNotFoundError) as error:
                    if 'imagecodecs' in str(error):
                        raise ValueError('This compressed TIFF requires imagecodecs. Stop the app and run: '+sys.executable+' -m pip install imagecodecs, then restart the app.') from error
                    raise
                if len(frames) > 100:
                    raise ValueError('A TIFF may contain at most 100 pages.')
    else:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError('Image exceeds 40 million pixels.')
            if im.mode == "P":
                im = im.convert("RGB")
            frames.append(np.asarray(im).copy());encodings.append((False,1.,0.,"native grayscale"))
    # Some acquisition TIFFs store the display LUT only on an adjacent 8-bit
    # preview, while the matching 16-bit sensor page says MINISBLACK. Reuse
    # orientation only for strongly matching, spatially aligned pixel planes.
    for i, raw in enumerate(frames):
        if (raw.ndim != 2 or raw.dtype != np.uint16
                or encodings[i][3] != "native grayscale"):
            continue
        matches = []
        for j in (i - 1, i + 1):
            if not 0 <= j < len(frames):
                continue
            preview = frames[j]
            if ('palette' not in encodings[j][3] or preview.dtype != np.uint8
                    or preview.shape != raw.shape):
                continue
            # Bounded deterministic sampling includes the entire spatial extent.
            step = max(1, int(math.ceil(raw.size / 262144)))
            x = raw.ravel()[::step].astype(float)
            y = preview.ravel()[::step].astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if abs(corr) >= .99:
                matches.append(bool(encodings[j][0]) != (corr < 0))
        if matches and len(set(matches)) == 1:
            encodings[i] = (matches[0], 1., 0.,
                            "native grayscale; orientation from matching adjacent preview LUT")
    result = []
    for page, raw in enumerate(frames):
        inverted,white,black,conversion=encodings[page]
        native_rgb = None
        if raw.ndim == 3 and raw.shape[-1] in (3, 4):
            if raw.shape[-1] == 4 and not np.all(raw[..., 3] == np.iinfo(raw.dtype).max):
                raise ValueError('Transparent images cannot be quantified. Provide opaque original images.')
            native_rgb = raw[..., :3].copy()
            native_rgb.setflags(write=False)
            raw_dtype = raw.dtype
            gray = np.dot(raw[..., :3].astype(np.float64), [0.299, 0.587, 0.114])
            conversion = "RGB luminance: 0.299R + 0.587G + 0.114B; no channel is silently selected"
        elif raw.ndim == 2:
            gray, raw_dtype = raw.copy(), raw.dtype
        else:
            raise ValueError(f"Page {page + 1}, shape {raw.shape}, is not a 2D grayscale/RGB image. Export channels and planes explicitly.")
        if gray.size > MAX_PIXELS or not np.isfinite(gray).all():
            raise ValueError('Pixel limit exceeded, or image contains NaN / Inf.')
        if np.issubdtype(raw_dtype, np.unsignedinteger):
            ceiling = float(np.iinfo(raw_dtype).max)
        else:
            raise ValueError('Quantification supports unsigned 8/16/32-bit images. Float or signed images require an explicitly defined acquisition range.')
        gray.setflags(write=False)
        identity = sha(data)[:16] + f"-p{page}"
        result.append(Source(identity, name, data, page, gray, ceiling,
                             display_white=ceiling, native_inverted=inverted,palette_white=white,palette_black=black,conversion=conversion, native_rgb=native_rgb))
    return result


def visual_dark_polarity(source: Source) -> str:
    """Raw-pixel signal direction corresponding to dark bands in the preview."""
    return "light" if source.native_inverted != source.display_invert else "dark"


def polarity_label(source: Source, polarity: str) -> str:
    return 'Dark bands / light background' if polarity == visual_dark_polarity(source) else 'Light bands / dark background'


def saturation(source: Source, polarity: str) -> np.ndarray:
    """Instrument limits, not observed image maximum. Dark film can clip at the lower end."""
    if source.ceiling <= source.floor:
        raise ValueError('Sensor maximum must exceed minimum.')
    data = source.native_rgb if source.native_rgb is not None else source.gray
    mask = data >= source.ceiling if polarity == "light" else data <= source.floor
    return np.any(mask, axis=-1) if mask.ndim == 3 else mask


def display_image(source: Source, overlay: np.ndarray | None = None) -> Image.Image:
    if source.display_white <= source.display_black:
        raise ValueError('Display white point must exceed black point.')
    z = (source.gray.astype(np.float32) - source.display_black) / (source.display_white - source.display_black)
    if source.native_inverted:z=1-z
    z=source.palette_black+z*(source.palette_white-source.palette_black)
    # One affine transform applied uniformly to the entire source. No gamma/local correction.
    z = np.clip((z - .5) * source.contrast + .5 + source.brightness, 0, 1)
    rgb = np.repeat(np.rint(z * 255).astype(np.uint8)[..., None], 3, axis=2)
    if source.display_invert:
        rgb = 255 - rgb
    if overlay is not None:
        rgb[overlay] = (255, 65, 65)
    return Image.fromarray(rgb)


def image_data_url(im: Image.Image, max_side=1400) -> str:
    im = im.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def rect_slice(rect: list[int], shape: tuple[int, int]) -> tuple[slice, slice]:
    if len(rect) != 4 or any(not isinstance(x, (int, np.integer)) for x in rect):
        raise ValueError('Regions must use integer x, y, width and height.')
    x, y, w, h = rect
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError(f"Region {rect} exceeds source-image bounds {shape}.")
    return slice(y, y + h), slice(x, x + w)


def overlap(a, b):
    return max(a[0], b[0]) < min(a[0]+a[2], b[0]+b[2]) and max(a[1], b[1]) < min(a[1]+a[3], b[1]+b[3])


@dataclass
class Protein:
    id: str
    name: str
    source_id: str
    membrane_id: str
    molecular_weight: str
    crop: list[int]
    signal: list[list[int]]
    background: list[list[int]]
    crop_selected: bool = False
    quant_roi_signature: str = ""
    not_detected_lanes: list[int] = field(default_factory=list)
    crop_angle: float = 0.0
    polarity: str = "dark"
    background_method: str = "mean"
    markers: list[dict] = field(default_factory=list)  # source y + label
    manual_lines: list[int] = field(default_factory=list)  # source x; vertical dividers
    segments: list[list[int]] = field(default_factory=list)  # [source x, width], same crop y/h


def default_protein(source: Source, lanes: int, name="Target") -> Protein:
    h, w = source.gray.shape
    if w < lanes * 3 or h < 8:
        raise ValueError('Source image is too small for this number of lanes.')
    left = int(w * .05)
    pitch = max(3, int(w * .9 / lanes))
    y, rh = int(h * .35), max(2, int(h * .10))
    bg_y, bg_h = min(h-2, int(h * .60)), max(1, int(h * .04))
    signal = [[left+i*pitch+int(pitch*.2), y, max(1,int(pitch*.6)), min(rh,h-y)] for i in range(lanes)]
    backgrounds = [[r[0], bg_y, r[2], min(bg_h, h-bg_y)] for r in signal]
    result=Protein(str(uuid.uuid4()), name, source.id, "", "", [left, max(0,y-10), min(w-left,pitch*lanes), min(h-max(0,y-10),rh+20)], signal, backgrounds)
    result.polarity=visual_dark_polarity(source)
    return result


def quant_roi_signature(protein: Protein) -> str:
    return sha(canonical({'source_id':protein.source_id,'signal':protein.signal,'background':protein.background,
                          'polarity':protein.polarity,'background_method':protein.background_method,
                          'crop':protein.crop,'crop_angle':protein.crop_angle}))


def measure(source: Source, protein: Protein, lane: int, exclude_saturated=True, subtract_background=True) -> dict:
    if source.ceiling <= source.floor:
        raise ValueError('Sensor maximum must exceed minimum.')
    signal_rect=protein.signal[lane]
    bg_rect=protein.background[lane] if subtract_background else None
    sig = source.gray[rect_slice(signal_rect, source.gray.shape)].astype(np.float64)
    bg = source.gray[rect_slice(bg_rect, source.gray.shape)].astype(np.float64) if subtract_background else None
    def local_saturation(rect):
        data = source.native_rgb if source.native_rgb is not None else source.gray
        block = data[rect_slice(rect,source.gray.shape)]
        mask = block >= source.ceiling if protein.polarity == "light" else block <= source.floor
        if mask.ndim == 3: mask = np.any(mask,axis=-1)
        return float(mask.mean())
    frac = local_saturation(signal_rect)
    bg_frac = local_saturation(bg_rect) if subtract_background else 0.
    transform = (lambda a: source.ceiling-a) if protein.polarity == "dark" else (lambda a: a-source.floor)
    raw = float(transform(sig).sum(dtype=np.float64))
    bg_value = float(np.median(transform(bg)) if protein.background_method == "median" else transform(bg).mean()) if subtract_background else 0.
    corrected = raw-bg_value*sig.size
    reasons = []
    if subtract_background and overlap(signal_rect, bg_rect):
        reasons.append("background_overlaps_signal")
    if exclude_saturated and (frac > 0 or bg_frac > 0):
        reasons.append("saturated_signal_or_background")
    if (sig < source.floor).any() or (sig > source.ceiling).any() or (subtract_background and ((bg < source.floor).any() or (bg > source.ceiling).any())):
        reasons.append("pixels_outside_sensor_range")
    return {"area_px": sig.size, "raw_integrated": raw, "background_signal": bg_value,
            "corrected_integrated": corrected, "band_intensity": corrected if subtract_background else raw,
            "background_subtracted": subtract_background, "background_integrated":bg_value*sig.size,
            "measurement_note":"net_signal_not_above_background" if subtract_background and corrected<=0 else "", "saturated_fraction": frac,
            "background_saturated_fraction": bg_frac, "valid": not reasons, "status": ";".join(reasons) or "ok"}


def quantify(sources: dict[str, Source], proteins: list[Protein], samples: list[dict], reference_id: str | None,
             control_condition: str | None = None, exclude_saturated=True, subtract_background=True, strict_replicates=True, require_confirmed=False) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    if any(not str(s.get('sample','')).strip() for s in samples):raise ValueError('Lane labels cannot be empty.')
    stats_valid=not any(not str(s.get(k,'')).strip() for s in samples for k in ('condition','bio_rep','tech_rep'))
    keys=[(s.get('condition',''),str(s.get('bio_rep','')),str(s.get('tech_rep',''))) for s in samples]
    stats_valid=stats_valid and len(set(keys))==len(keys)
    if strict_replicates and not stats_valid:raise ValueError('Sample, condition and replicate IDs must be filled; biological/technical replicate combinations must be unique within each condition.')
    if len({p.id for p in proteins}) != len(proteins):
        raise ValueError('Duplicate protein IDs.')
    ref = next((p for p in proteins if p.id == reference_id), None)
    if reference_id and ref is None:
        raise ValueError('Selected loading control does not exist.')
    measured = {}
    for p in proteins:
        if len(p.signal) != len(samples) or (subtract_background and len(p.background) != len(samples)):
            raise ValueError('Lane count does not match samples. Regenerate regions or restore the sample count.')
        for lane in range(len(samples)):
            try:
                if require_confirmed and p.quant_roi_signature!=quant_roi_signature(p):
                    raise ValueError('quantification_rois_not_confirmed')
                measured[(p.id,lane)] = measure(sources[p.source_id],p,lane,exclude_saturated,subtract_background)
            except (ValueError,IndexError) as error:
                measured[(p.id,lane)] = {"valid":False,"status":str(error),"band_intensity":np.nan,"raw_integrated":np.nan,"background_integrated":np.nan,"measurement_note":""}
    for p in proteins:
        for lane in range(len(samples)):
            m=measured[(p.id,lane)]
            marked=lane in p.not_detected_lanes
            m['not_detected_manual']=marked
            m['zero_override_applied']=bool(marked and m.get('valid'))
            m['analysis_intensity']=0. if m['zero_override_applied'] else (m.get('band_intensity',np.nan) if m.get('valid') else np.nan)
            m['zero_assignment_source']='user_marked_no_visible_band' if marked else ''
    records = []
    for p in proteins:
        for lane,sample in enumerate(samples):
            m = measured[(p.id,lane)].copy()
            ratio = np.nan
            issues = [] if m.get("valid") else [m.get("status", "invalid")]
            r = measured.get((ref.id,lane)) if ref else None
            if ref:
                if not p.membrane_id.strip() or p.membrane_id != ref.membrane_id:
                    issues.append("membrane_pairing_unconfirmed_or_mismatch")
                elif not r.get("valid") or r.get("analysis_intensity",0) <= 0:
                    issues.append("invalid_or_nonpositive_reference")
                elif m.get("valid"):
                    ratio = m["analysis_intensity"]/r["analysis_intensity"]
            analysis_value = ratio if ref else (m.get("analysis_intensity",np.nan) if m.get("valid") else np.nan)
            records.append({"protein_id":p.id,"protein":p.name,"lane":lane+1,**sample,**m,
                            "source_id":p.source_id,"source_sha256":sha(sources[p.source_id].original),
                            "source_page":sources[p.source_id].page,"membrane_id":p.membrane_id,
                            "reference":ref.name if ref else "", "reference_intensity":r.get("analysis_intensity",np.nan) if r else np.nan, "reference_measured_intensity":r.get("band_intensity",np.nan) if r else np.nan, "reference_not_detected_manual":bool(lane in ref.not_detected_lanes) if ref else False, "normalized_ratio":ratio,"analysis_value":analysis_value,
                            "analysis_status":";".join(issues) or "ok",
                            "signal_xywh":json.dumps(p.signal[lane]),"background_xywh":json.dumps(p.background[lane]) if subtract_background else "",
                            "background_method":p.background_method if subtract_background else "none","polarity":p.polarity})
    raw = pd.DataFrame(records)
    # Technical replicates collapse first; SD/n count independent biological replicates.
    bio = raw.groupby(["protein_id","protein","condition","bio_rep"],dropna=False,sort=False)["analysis_value"].agg(
        technical_n="count", bio_value="mean").reset_index() if stats_valid else pd.DataFrame(columns=["protein_id","protein","condition","bio_rep","technical_n","bio_value"])
    control_means = bio[bio.condition == control_condition].groupby("protein_id").bio_value.mean() if control_condition else pd.Series(dtype=float)
    raw["fold_to_control"] = [v / control_means.get(pid,np.nan) if control_means.get(pid,0)>0 else np.nan for pid,v in zip(raw.protein_id,raw.analysis_value)]
    bio["fold_to_control"] = [v / control_means.get(pid,np.nan) if control_means.get(pid,0)>0 else np.nan for pid,v in zip(bio.protein_id,bio.bio_value)]
    summary = bio.groupby(["protein_id","protein","condition"],dropna=False,sort=False).agg(
        biological_n=("bio_value","count"),mean=("bio_value","mean"),sd=("bio_value","std"),
        fold_mean=("fold_to_control","mean"),fold_sd=("fold_to_control","std")).reset_index()
    summary["reference"] = ref.name if ref else ""
    return raw,bio,summary


def get_font(size: int):
    candidates = [Path(__file__).resolve().parent.parent / "assets" / "font.ttf",
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                  Path("/System/Library/Fonts/Supplemental/Arial.ttf"),Path("C:/Windows/Fonts/arial.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def figure(sources: dict[str,Source], proteins: list[Protein], samples: list[dict], *, dpi=300,
           lane_width_mm=7.0, font_pt=8.0, gap_mm=2.0, label_margin_mm=22.0) -> tuple[Image.Image,dict]:
    if dpi not in (300,600):
        raise ValueError('DPI must be 300 or 600.')
    unit=dpi/25.4
    width=max(1,round(len(samples)*lane_width_mm*unit))
    margin=round(label_margin_mm*unit)
    font=get_font(max(8,round(font_pt*dpi/72)))
    gap=round(gap_mm*unit)
    # Reserve enough vertical space for protein and kDa labels. Scale all source pixels equally.
    top=round(font_pt*dpi/72*2.5)
    rows=[]
    display_cache={}
    for p in proteins:
        source=sources[p.source_id]
        rect_slice(p.crop,source.gray.shape)
        x,y,w,h=p.crop
        segments=p.segments or [[x,w]]
        previous_end=None
        for xx,ww in segments:
            rect_slice([xx,y,ww,h],source.gray.shape)
            if xx<x or xx+ww>x+w or (previous_end is not None and xx<previous_end):
                raise ValueError('Splice segments must be inside the crop, ordered left to right without overlap.')
            previous_end=xx+ww
        joined=Image.new('RGB',(sum(ww for xx,ww in segments),h),'white')
        if source.id not in display_cache:
            display_cache[source.id]=display_image(source)
        source_display=display_cache[source.id]
        offset=0
        offsets=[]
        for xx,ww in segments:
            joined.paste(source_display.crop((xx,y,xx+ww,y+h)),(offset,0))
            offsets.append((xx,ww,offset))
            offset+=ww
        scale=width/joined.width
        height=max(1,round(h*scale))
        # OpenCV area downsampling / cubic upsampling affect display only, never quantification.
        resized=cv2.resize(np.asarray(joined),(width,height),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        row_extent=max(height,round(font_pt*dpi/72*2.8))
        rows.append({"p":p,"image":Image.fromarray(resized),"scale":scale,"offsets":offsets,
                     "height":height,"extent":row_extent,"upsampled":scale>1.001})
    height=top+sum(r['extent'] for r in rows)+gap*max(0,len(rows)-1)
    if (width+margin)*height > MAX_PIXELS:
        raise ValueError('Figure exceeds 40 million pixels. Reduce panel size or use 300 DPI.')
    result=Image.new('RGB',(margin+width,height),'white')
    draw=ImageDraw.Draw(result)
    row_y=top
    expected_lane_centers=None
    for ri,r in enumerate(rows):
        p=r['p'];x,y,w,h=p.crop
        if max(font.getlength(p.name),font.getlength(p.molecular_weight)) > margin-12:
            raise ValueError('Protein label is too long. Increase the left margin or reduce font size.')
        image_y=row_y+(r['extent']-r['height'])//2
        result.paste(r['image'],(margin,image_y))
        # Independent strips are disclosed by mandatory borders plus white space.
        draw.rectangle((margin,image_y,margin+width-1,image_y+r['height']-1),outline='#777777',width=max(1,round(dpi/300)))
        draw.text((margin-8,row_y+r['extent']/2- (font.size*.65 if p.molecular_weight else 0)),p.name,font=font,fill='black',anchor='rm')
        if p.molecular_weight:
            draw.text((margin-8,row_y+r['extent']/2+font.size*.65),p.molecular_weight,font=font,fill='black',anchor='rm')
        def map_x(source_x):
            for xx,ww,offset in r['offsets']:
                if xx <= source_x < xx+ww:
                    return margin+(offset+source_x-xx)*r['scale']
            return None
        centers=[map_x(roi[0]+roi[2]/2) for roi in p.signal]
        if expected_lane_centers is None:
            expected_lane_centers=centers
        else:
            for expected,actual in zip(expected_lane_centers,centers):
                if (expected is None) != (actual is None) or (expected is not None and abs(expected-actual)>width/len(samples)*.12):
                    raise ValueError('Sample lanes are not aligned between proteins. Check crops, lane regions and splice segments.')
        # Top labels are shared only after validating lane mapping across all strips.
        for lane,sample in enumerate(samples):
            roi=p.signal[lane]
            center=map_x(roi[0]+roi[2]/2)
            if center is not None:
                if font.getlength(sample['sample']) > width/len(samples)*.95:
                    raise ValueError('Sample label exceeds lane width. Shorten the label, increase lane width or reduce font size.')
                if ri==0:
                    draw.text((center,top-font.size*.55),sample['sample'],font=font,fill='black',anchor='mb')
        # Auto-dividers mark every separate source segment, even if user requests contiguous joins.
        for xx,ww,offset in r['offsets'][1:]:
            xx_out=margin+offset*r['scale']
            draw.line((xx_out,image_y,xx_out,image_y+r['height']-1),fill='black',width=max(2,round(dpi/150)))
        for source_x in p.manual_lines:
            xx_out=map_x(source_x)
            if xx_out is not None:
                draw.line((xx_out,image_y,xx_out,image_y+r['height']-1),fill='black',width=max(2,round(dpi/150)))
        for marker in p.markers:
            yy=image_y+(float(marker['y'])-y)*r['scale']
            if image_y <= yy < image_y+r['height']:
                draw.line((margin,yy,margin+round(1.2*unit),yy),fill='#111',width=max(1,round(dpi/300)))
                draw.text((margin+round(1.5*unit),yy),str(marker['label']),font=font,fill='black',anchor='lm')
        r['output_y']=image_y
        row_y+=r['extent']+gap
    metadata={"dpi":dpi,"pixels":list(result.size),"physical_mm":[result.width/unit,result.height/unit],
              "lane_width_mm":lane_width_mm,"font_pt":font_pt,"interpolation":"area downsample / cubic upsample (display only)",
              "strips":[{"protein":r['p'].name,"source_id":r['p'].source_id,"crop":r['p'].crop,
                         "segments":r['p'].segments,"upsampled":r['upsampled'],"output_y":r['output_y'],"scale":r['scale']} for r in rows]}
    return result,metadata


def export_tiff(im: Image.Image,dpi: int) -> bytes:
    buf=io.BytesIO()
    # Presentation TIFF is 8-bit RGB; raw TIFFs keep their native 16-bit samples in project/originals.
    im.save(buf,format='TIFF',compression='tiff_lzw',dpi=(dpi,dpi))
    return buf.getvalue()


def export_pdf(im: Image.Image,dpi: int) -> bytes:
    buf=io.BytesIO()
    width,height=im.width/dpi*72,im.height/dpi*72
    pdf=pdfcanvas.Canvas(buf,pagesize=(width,height))
    pdf.setTitle('Western blot figure panel')
    pdf.drawImage(ImageReader(im),0,0,width,height,mask='auto')
    pdf.showPage();pdf.save()
    return buf.getvalue()


def export_excel(raw: pd.DataFrame,bio: pd.DataFrame,summary: pd.DataFrame,details: pd.DataFrame | None=None) -> bytes:
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        frames=[('Raw_and_normalized',raw),('Biological_replicates',bio),('Mean_SD',summary)]
        if details is not None:frames.append(('All_protein_intensities',details))
        for name,frame in frames:
            safe=frame.copy()
            for col in safe.select_dtypes(include=['object','string']):
                safe[col]=safe[col].map(lambda v: "'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
            safe.to_excel(writer,sheet_name=name,index=False)
    return buf.getvalue()


def safe_csv(frame):
    safe=frame.copy()
    for col in safe.select_dtypes(include=['object','string']):
        safe[col]=safe[col].map(lambda v: "'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v)
    return safe.to_csv(index=False).encode('utf-8-sig')


def project_bytes(sources: dict[str,Source],proteins: list[Protein],samples: list[dict],audit: Audit,settings: dict,
                  artifacts: dict[str,bytes] | None=None) -> bytes:
    buf=io.BytesIO()
    manifest={"version":VERSION,"proteins":[asdict(p) for p in proteins],"samples":samples,"settings":settings,
              "sources":[s.metadata() for s in sources.values()],"session":audit.session}
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('project.json',canonical(manifest))
        z.writestr('audit.json',canonical(audit.document(settings,manifest['sources'])))
        written=set()
        for s in sources.values():
            path='originals/'+sha(s.original)+Path(s.name).suffix.lower()
            if path not in written:
                z.writestr(path,s.original);written.add(path)
        for name,data in (artifacts or {}).items():
            z.writestr('exports/'+Path(name).name,data)
    return buf.getvalue()


def restore_project(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(i.file_size for i in z.infolist())>500*1024*1024 or len(z.infolist())>500:
            raise ValueError('Unpacked project exceeds size limits.')
        m=json.loads(z.read('project.json'));a=json.loads(z.read('audit.json'))
        if m['version'] != VERSION:
            raise ValueError('Unsupported project version.')
        audit=Audit(m['session'],a['events'])
        if not audit.verify():
            raise ValueError('Processing history chain is incomplete or modified.')
        sources={};decoded={}
        for info in m['sources']:
            digest=info['original_sha256'];suffix=Path(info['name']).suffix.lower()
            key=(digest,suffix)
            if key not in decoded:
                original=z.read('originals/'+digest+suffix)
                if sha(original)!=digest:
                    raise ValueError('Original image hash mismatch.')
                decoded[key]=load_sources(original,info['name'])
            source=decoded[key][info['page']]
            if source.id!=info['id'] or sha(source.gray.tobytes())!=info['gray_sha256']:
                raise ValueError('Decoded pixel hash mismatch.')
            for prop in ('ceiling','floor','display_black','display_white','brightness','contrast'):
                setattr(source,prop,info['sensor_'+prop] if prop in ('ceiling','floor') else info[prop])
            source.display_invert=bool(info.get("display_invert",False))
            # New decoding may recover orientation previously omitted from a
            # 16-bit page. Preserve saved appearance and all raw quantification.
            if bool(info.get("native_inverted", False)) != source.native_inverted:
                source.display_invert=not source.display_invert
            sources[source.id]=source
        proteins=[Protein(**p) for p in m['proteins']]
        if not 1 <= len(m['samples']) <= 40:
            raise ValueError('Invalid protein or sample count.')
        for p in proteins:
            if p.source_id not in sources or len(p.signal)!=len(m['samples']) or len(p.background)!=len(m['samples']):
                raise ValueError('Project regions do not match lane count.')
            rect_slice(p.crop,sources[p.source_id].gray.shape)
        audit.add('restore_project',{'project_sha256':sha(data),'restored_event_count':len(audit.events)})
        from .annotations import migrate_font_settings
        return sources,proteins,m['samples'],audit,migrate_font_settings(m['settings'])
