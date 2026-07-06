"""Interactive HTML renderer.

Wraps the SVG output in an HTML shell that adds:

- **Pan / zoom** via a small inline JS implementation (no external CDN).
- **Click-to-inspect**: clicking any node populates a side panel with
  the layer type, params, shapes, and attributes. The graph data is
  serialized once into ``<script id="mv-graph" type="application/json">``
  so the panel doesn't need per-node listeners on every re-render.
- **Keyboard shortcuts**: ``+`` / ``-`` zoom, ``r`` reset, ``f`` fit.

For M4 we ship this dependency-free — a future advanced mode could add
D3.js features (per PRD §13 Q2 we chose lightweight over D3).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from modelvision.core.style import Group, NodeStyle, Theme
from modelvision.renderers.svg_renderer import render_svg

if TYPE_CHECKING:
    from modelvision.layout import LaidOutGraph


_PAN_ZOOM_JS = """
(function () {
  const svg = document.querySelector('#mv-svg svg');
  if (!svg) return;
  const viewBox = svg.viewBox.baseVal;
  const state = {
    minX: viewBox.x, minY: viewBox.y,
    w: viewBox.width, h: viewBox.height,
    origW: viewBox.width, origH: viewBox.height,
    origX: viewBox.x, origY: viewBox.y,
  };
  function apply() {
    svg.setAttribute('viewBox', `${state.minX} ${state.minY} ${state.w} ${state.h}`);
  }
  function zoom(factor, cx, cy) {
    const pt = svg.createSVGPoint();
    pt.x = cx; pt.y = cy;
    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
    state.minX = svgP.x - (svgP.x - state.minX) / factor;
    state.minY = svgP.y - (svgP.y - state.minY) / factor;
    state.w /= factor; state.h /= factor;
    apply();
  }
  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    zoom(factor, e.clientX, e.clientY);
  }, { passive: false });

  let dragging = false, startX = 0, startY = 0, startMinX = 0, startMinY = 0;
  svg.addEventListener('mousedown', (e) => {
    dragging = true;
    startX = e.clientX; startY = e.clientY;
    startMinX = state.minX; startMinY = state.minY;
    svg.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const rect = svg.getBoundingClientRect();
    const scaleX = state.w / rect.width, scaleY = state.h / rect.height;
    state.minX = startMinX - (e.clientX - startX) * scaleX;
    state.minY = startMinY - (e.clientY - startY) * scaleY;
    apply();
  });
  window.addEventListener('mouseup', () => { dragging = false; svg.style.cursor = ''; });

  window.addEventListener('keydown', (e) => {
    if (e.key === '+' || e.key === '=') { zoom(1.2, innerWidth / 2, innerHeight / 2); }
    else if (e.key === '-') { zoom(1 / 1.2, innerWidth / 2, innerHeight / 2); }
    else if (e.key === 'r' || e.key === 'f') {
      state.minX = state.origX; state.minY = state.origY;
      state.w = state.origW; state.h = state.origH;
      apply();
    }
  });
})();
"""

_INSPECT_JS = """
(function () {
  const data = JSON.parse(document.getElementById('mv-graph').textContent);
  const nodesById = Object.fromEntries(data.nodes.map(n => [n.id, n]));
  const panel = document.getElementById('mv-inspector');
  document.querySelectorAll('#mv-svg [data-node-id]').forEach((el) => {
    el.style.cursor = 'pointer';
    el.addEventListener('click', (e) => {
      e.stopPropagation();
      const id = el.getAttribute('data-node-id');
      const n = nodesById[id];
      if (!n) return;
      const attrs = Object.entries(n.attributes || {})
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => `<tr><th>${k}</th><td>${Array.isArray(v) ? v.join('×') : v}</td></tr>`)
        .join('');
      panel.innerHTML = `
        <h2>${n.name || n.id}</h2>
        <p class="mv-type">${n.layer_type}</p>
        <table>
          <tr><th>id</th><td>${n.id}</td></tr>
          ${n.params ? `<tr><th>params</th><td>${n.params.toLocaleString()}</td></tr>` : ''}
          ${n.input_shape ? `<tr><th>input</th><td>${JSON.stringify(n.input_shape)}</td></tr>` : ''}
          ${n.output_shape ? `<tr><th>output</th><td>${JSON.stringify(n.output_shape)}</td></tr>` : ''}
          ${attrs}
        </table>`;
      panel.classList.add('mv-open');
    });
  });
  document.getElementById('mv-svg').addEventListener('click', () => {
    panel.classList.remove('mv-open');
  });
})();
"""

_CSS = """
:root { color-scheme: light dark; }
body { margin: 0; font-family: Inter, system-ui, sans-serif; background: #f7f7fa; color: #1a1a1a; }
header { padding: 12px 20px; background: #1a1a2e; color: #f5f5fa; font-weight: 600; display: flex; justify-content: space-between; }
header small { font-weight: 400; opacity: 0.7; }
#mv-svg { width: 100vw; height: calc(100vh - 48px); overflow: hidden; }
#mv-svg svg { width: 100%; height: 100%; display: block; cursor: grab; }
#mv-inspector {
  position: fixed; right: 0; top: 48px; height: calc(100vh - 48px);
  width: 320px; background: rgba(255,255,255,0.95); border-left: 1px solid #ddd;
  padding: 20px; overflow-y: auto; box-shadow: -2px 0 12px rgba(0,0,0,0.05);
  transform: translateX(100%); transition: transform 0.2s ease-out;
}
#mv-inspector.mv-open { transform: translateX(0); }
#mv-inspector h2 { margin: 0 0 4px; font-size: 16px; }
#mv-inspector .mv-type { color: #6b7280; margin: 0 0 12px; font-size: 13px; }
#mv-inspector table { width: 100%; border-collapse: collapse; font-size: 13px; }
#mv-inspector th { text-align: left; padding: 4px 8px 4px 0; color: #6b7280; font-weight: 500; vertical-align: top; }
#mv-inspector td { padding: 4px 0; font-family: ui-monospace, monospace; word-break: break-word; }
.mv-hint { position: fixed; bottom: 12px; left: 12px; font-size: 11px; color: #6b7280; }
"""


def render_html(
    laid_out: LaidOutGraph,
    *,
    theme: Theme,
    layer_palette: dict[str, str] | None = None,
    groups: list[Group] | None = None,
    node_styles: dict[str, NodeStyle] | None = None,
    show_params: bool = True,
    show_shapes: bool = True,
    show_dtypes: bool = False,
    embed_fonts: bool = True,
    title: str | None = None,
    legend: bool = False,
    default_shape: str | None = None,
    flow_style: bool = False,
) -> str:
    """Return a full HTML document with an interactive SVG diagram."""
    svg = render_svg(
        laid_out,
        theme=theme,
        layer_palette=layer_palette,
        groups=groups,
        node_styles=node_styles,
        show_params=show_params,
        show_shapes=show_shapes,
        show_dtypes=show_dtypes,
        embed_fonts=embed_fonts,
        title=title,
        legend=legend,
        default_shape=default_shape,
        flow_style=flow_style,
    )
    graph_json = json.dumps(laid_out.graph.to_dict(), default=str)
    doc_title = title or laid_out.graph.metadata.get("model_class", "ModelVision")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{doc_title}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <span>{doc_title}</span>
  <small>ModelVision · scroll to zoom · drag to pan · +/- to zoom · r/f to reset</small>
</header>
<div id="mv-svg">{svg}</div>
<aside id="mv-inspector"></aside>
<script id="mv-graph" type="application/json">{graph_json}</script>
<script>{_PAN_ZOOM_JS}</script>
<script>{_INSPECT_JS}</script>
</body>
</html>
"""


__all__ = ["render_html"]
