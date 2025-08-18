# svg.py
# simple SVG writer for paths/overlays

def path_d(points):
    if not points: return ""
    d=f"M {points[0][0]} {points[0][1]}"
    for x,y in points[1:]: d+=f" L {x} {y}"
    return d+" Z"

def write_svg(paths, size, out_path, hole_stroke=False):
    w,h=size
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<g fill="none" stroke="red" stroke-width="1">']
    for outer, holes in paths:
        parts.append(f'<path d="{path_d(outer)}" />')
        for hc in holes:
            parts.append(f'<path d="{path_d(hc)}" stroke="blue" />' if hole_stroke else f'<path d="{path_d(hc)}" />')
    parts.append('</g></svg>')
    with open(out_path,'w') as f: f.write("\n".join(parts))
