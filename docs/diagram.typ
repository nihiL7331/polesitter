#set page(width: auto, height: auto, margin: 16pt, fill: none)
#set text(
  font: "BigBlueTerm437 Nerd Font Mono",
  fill: rgb("#C9D1D9"),
  size: 14pt,
)

#let col-up = rgb("#1F6FEB") 
#let col-down = rgb("#238636")
#let col-inter = rgb("#8957E5")
#let col-direct = rgb("#D29922")
#let col-bg = rgb("#21262D")
#let col-border = rgb("#8B949E")
#let col-ptr = rgb("#A5D6FF")

#let flow-block(w, bg, label, subtext: "") = rect(
  width: w,
  height: 50pt,
  fill: bg,
  stroke: 1pt + col-border,
  radius: 4pt,
  align(center + horizon)[
    #text(fill: white, weight: "bold", label)
    #if subtext != "" [ \ #text(fill: col-ptr, size: 8pt, subtext) ]
  ]
)

#let arrow-down = align(center)[#text(fill: col-border, size: 18pt, weight: "bold")[ ↓ ]]

#let bw = 300pt

#let barnes-hut = stack(
  dir: ttb,
  spacing: 12pt,
  align(center)[#text(weight: "bold", size: 16pt)[Barnes-Hut (O(N log N))]],
  line(length: bw, stroke: col-border),
  
  flow-block(bw, col-up, "upward pass", subtext: "P2M & M2M (leaves -> root)"),
  arrow-down,
  flow-block(bw, col-direct, "evaluation pass", subtext: "tree traversal + MAC -> P2M / P2P"),
  
  rect(width: bw, height: 62pt, stroke: none),
  rect(width: bw, height: 62pt, stroke: none)
)

#let fmm = stack(
  dir: ttb,
  spacing: 12pt,
  align(center)[#text(weight: "bold", size: 16pt)[FMM (O(N))]],
  line(length: bw, stroke: col-border),
  
  flow-block(bw, col-up, "upward pass", subtext: "P2M & M2M (leaves -> root)"),
  arrow-down,
  flow-block(bw, col-inter, "interaction pass", subtext: "M2L (cell <-> cell)"),
  arrow-down,
  flow-block(bw, col-down, "downward pass", subtext: "L2L (root -> leaves)"),
  arrow-down,
  
  stack(
    dir: ltr,
    spacing: 10pt,
    flow-block(bw/2 - 5pt, col-down, "4a. far field", subtext: "L2P (local -> part)"),
    flow-block(bw/2 - 5pt, col-direct, "4b. near field", subtext: "P2P")
  )
)

#grid(
  columns: (bw, bw),
  column-gutter: 40pt,
  barnes-hut,
  fmm
)
