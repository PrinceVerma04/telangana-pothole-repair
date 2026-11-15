/** Field photos: downscale to <= 1280 px on the long side and re-encode as JPEG before queueing/uploading. */
export const MAX_PHOTO_PX = 1280;

/** Target size that fits inside max x max while keeping aspect ratio; never upscales. */
export function fitWithin(w: number, h: number, max = MAX_PHOTO_PX): { w: number; h: number; scale: number } {
  if (w <= 0 || h <= 0) return { w: 0, h: 0, scale: 0 };
  const scale = Math.min(1, max / Math.max(w, h));
  return { w: Math.max(1, Math.round(w * scale)), h: Math.max(1, Math.round(h * scale)), scale };
}

export interface ResizedPhoto { blob: Blob; width: number; height: number; originalBytes: number }

/**
 * Decode (EXIF orientation applied by createImageBitmap), resize and encode to JPEG.
 * Uses OffscreenCanvas when available, otherwise a DOM canvas.
 */
export async function resizePhoto(file: Blob, max = MAX_PHOTO_PX, quality = 0.8): Promise<ResizedPhoto> {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' } as ImageBitmapOptions);
  const t = fitWithin(bmp.width, bmp.height, max);
  let blob: Blob;
  if (typeof OffscreenCanvas !== 'undefined') {
    const c = new OffscreenCanvas(t.w, t.h);
    const ctx = c.getContext('2d');
    if (!ctx) throw new Error('2d canvas unavailable');
    ctx.drawImage(bmp, 0, 0, t.w, t.h);
    blob = await c.convertToBlob({ type: 'image/jpeg', quality });
  } else {
    const c = document.createElement('canvas');
    c.width = t.w;
    c.height = t.h;
    const ctx = c.getContext('2d');
    if (!ctx) throw new Error('2d canvas unavailable');
    ctx.drawImage(bmp, 0, 0, t.w, t.h);
    blob = await new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error('encode failed'))), 'image/jpeg', quality));
  }
  bmp.close?.();
  return { blob, width: t.w, height: t.h, originalBytes: file.size };
}
