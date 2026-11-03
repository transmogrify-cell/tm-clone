import { useEffect, useRef, useState, type MouseEvent } from 'react';

export interface LineSeries {
  name: string;
  color: string;
  values: (number | undefined)[];
}

interface LineChartProps {
  title: string;
  series: LineSeries[];
  /** Total epochs (x axis runs 1..xMax so the chart doesn't rescale as training proceeds). */
  xMax: number;
  yMin?: number;
  /** Fixed top (e.g. 1 for accuracy); otherwise a rounded-up max of the data. */
  yMax?: number;
  format(v: number): string;
  height?: number;
  /** Optional vertical marker (e.g. the best epoch). */
  marker?: { x: number; label: string };
}

const M = { l: 44, r: 84, t: 10, b: 26 };

function niceCeil(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((c) => c >= v) ?? 10 * p;
}

/**
 * Small responsive SVG line chart: one y axis, 2px lines, recessive grid,
 * legend + direct end labels, crosshair tooltip on hover.
 */
export function LineChart({ title, series, xMax, yMin = 0, yMax, format, height = 200, marker }: LineChartProps) {
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== undefined && Number.isFinite(v)));
  const top = yMax ?? niceCeil(Math.max(...all, 0.0001));
  const n = Math.max(2, xMax);
  const x = (i: number) => M.l + (i / (n - 1)) * (width - M.l - M.r); // i is 0-based epoch index
  const y = (v: number) => M.t + (1 - (v - yMin) / (top - yMin)) * (height - M.t - M.b);
  const ticks = [0, 1, 2, 3, 4].map((k) => yMin + ((top - yMin) * k) / 4);
  const xTicks = Array.from(new Set([1, Math.round(n / 4), Math.round(n / 2), Math.round((3 * n) / 4), n])).filter((t) => t >= 1);
  const longest = Math.max(0, ...series.map((s) => s.values.length));

  const path = (values: (number | undefined)[]) =>
    values
      .map((v, i) => (v === undefined || !Number.isFinite(v) ? null : `${x(i).toFixed(1)},${y(Math.min(v, top)).toFixed(1)}`))
      .reduce<string>((d, p, i, arr) => (p ? d + (d && arr[i - 1] ? 'L' : 'M') + p : d), '');

  // End labels, nudged apart when they would overlap.
  const ends = series
    .map((s) => {
      const i = s.values.length - 1;
      const v = s.values[i];
      return v === undefined || !Number.isFinite(v) ? null : { name: s.name, color: s.color, i, yy: y(Math.min(v, top)) };
    })
    .filter((e): e is NonNullable<typeof e> => e !== null)
    .sort((a, b) => a.yy - b.yy);
  for (let k = 1; k < ends.length; k++) if (ends[k].yy - ends[k - 1].yy < 14) ends[k].yy = ends[k - 1].yy + 14;

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left + M.l;
    const i = Math.round(((px - M.l) / (width - M.l - M.r)) * (n - 1));
    setHover(i >= 0 && i < longest ? i : null);
  };

  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-semibold text-slate-800">{title}</span>
        <span className="flex gap-3 text-xs text-slate-600">
          {series.map((s) => (
            <span key={s.name} className="flex items-center gap-1.5">
              <span className="h-0.5 w-3 rounded" style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </span>
      </figcaption>
      <div ref={wrap} className="relative">
        <svg width={width} height={height} role="img" aria-label={`${title} by epoch`} className="block overflow-visible">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={M.l} x2={width - M.r} y1={y(t)} y2={y(t)} stroke="#e7e5e4" strokeWidth={1} />
              <text x={M.l - 6} y={y(t)} dy="0.32em" textAnchor="end" className="fill-slate-400 text-[10px] tabular-nums">
                {format(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={x(t - 1)} y={height - 8} textAnchor="middle" className="fill-slate-400 text-[10px] tabular-nums">
              {t}
            </text>
          ))}
          <text x={width - M.r} y={height - 8} dx={14} className="fill-slate-400 text-[10px]">
            epoch
          </text>
          {marker && marker.x >= 1 && (
            <g>
              <line x1={x(marker.x - 1)} x2={x(marker.x - 1)} y1={M.t} y2={height - M.b} stroke="#a8a29e" strokeDasharray="3 3" />
              <text x={x(marker.x - 1)} y={M.t} dx={4} dy="0.8em" className="fill-slate-500 text-[10px]">
                {marker.label}
              </text>
            </g>
          )}
          {series.map((s) => (
            <path key={s.name} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {ends.map((e) => (
            <g key={e.name}>
              <circle cx={x(e.i)} cy={y(Math.min(series.find((s) => s.name === e.name)!.values[e.i]!, top))} r={3} fill={e.color} stroke="white" strokeWidth={2} />
              <text x={width - M.r + 8} y={e.yy} dy="0.32em" className="fill-slate-600 text-[11px]">
                {e.name}
              </text>
            </g>
          ))}
          {hover !== null && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={M.t} y2={height - M.b} stroke="#78716c" strokeWidth={1} />
              {series.map((s) => {
                const v = s.values[hover];
                return v === undefined || !Number.isFinite(v) ? null : (
                  <circle key={s.name} cx={x(hover)} cy={y(Math.min(v, top))} r={4.5} fill={s.color} stroke="white" strokeWidth={2} />
                );
              })}
            </g>
          )}
          <rect
            x={M.l}
            y={M.t}
            width={Math.max(0, width - M.l - M.r)}
            height={height - M.t - M.b}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
        </svg>
        {hover !== null && (
          <div
            className="pointer-events-none absolute z-10 rounded-lg bg-slate-900 px-2.5 py-1.5 text-xs text-white shadow-lg"
            style={{ left: Math.min(x(hover) + 10, width - 150), top: M.t }}
          >
            <div className="mb-0.5 font-semibold">Epoch {hover + 1}</div>
            {series.map((s) => (
              <div key={s.name} className="flex items-center gap-1.5 tabular-nums">
                <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                {s.name}: {s.values[hover] === undefined ? '–' : format(s.values[hover]!)}
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
