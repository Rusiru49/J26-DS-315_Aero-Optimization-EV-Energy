import Plotly from "plotly.js-dist-min";
import { useEffect, useRef } from "react";

const INK = "#16344d";
const GRID = "#e6eef4";

interface Props {
  data: unknown[];
  layout?: Record<string, unknown>;
  /** pixels, or leave empty to fill the panel that holds the chart */
  height?: number;
}

/** Thin wrapper around Plotly with the AeroEVLanka look (light, calm). */
export default function PlotlyChart({ data, layout = {}, height }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const l = layout as { xaxis?: object; yaxis?: object; margin?: object };
    Plotly.react(
      el,
      data,
      {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "Segoe UI, system-ui, sans-serif", color: INK, size: 11 },
        legend: { orientation: "h", y: 1.2, x: 0 },
        ...layout,
        margin: { l: 48, r: 10, t: 18, b: 32, ...l.margin },
        xaxis: { gridcolor: GRID, zerolinecolor: "#cfdce6", ...l.xaxis },
        yaxis: { gridcolor: GRID, zerolinecolor: "#cfdce6", ...l.yaxis },
        autosize: true,
      },
      { displayModeBar: false, responsive: true },
    );
  }, [data, layout]);

  // follow the size of the panel
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      if (el.clientWidth > 0 && el.clientHeight > 0) Plotly.Plots.resize(el);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} style={{ width: "100%", height: height ?? "100%", minHeight: 100 }} />;
}