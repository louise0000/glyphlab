# The Problem with Autotrace - Why We Need Stroke Skeleton Detection
## This movie has an audio soundtrack ✒️
https://github.com/user-attachments/assets/c9526f97-9cf6-4598-9ebe-cdfa4f770a7a



# Initial Logic

Before I could define an ML / DL problem, I needed to figure out what kinds of logic are at work when me—a human, approaches type-tracing in a very manual way through vector graphics software.

Breaking down different types of letter and logoform types into categories helped me design a decision tree, and arrange the forms along a spectrum as seen at the top of the tree here:

<img width="1209" height="558" alt="Screenshot 2025-08-18 at 9 36 02 pm" src="https://github.com/user-attachments/assets/15c12508-04c6-4465-9649-8b4401d5b48d" />

In this github repository is a vibe coded app, that uses traditional image processing algorithms from the 90s, and achieves good classification according to my decision tree rules just by checking things such as "is there a empty space completely enclosed by filled pixels?" if so, there is a compound shape (a path within a path). Is the area of fill significantly bigger than the compound windows? Then it's probably an image rather than a concentric path with a compound window. Are there junctions, indicated by a cluster of pixels with more than two protrusions...? It even attempts a voronoi skeleton producing a very sketchy attempt at an inner path for strokes. However, training using data such as can be generated below as well as a creative loss function need designing to improve upon the voronoi skeletons. 

# Initial Thoughts on Intersections
<img width="490" height="467" alt="Screenshot 2025-08-18 at 10 55 04 pm" src="https://github.com/user-attachments/assets/f2d842a1-0a41-4ff1-bb49-584919edbe2f" />

# Tools for Data Capture
[https://openprocessing.org/sketch/2703444]

↑Something like this could be used to generate thousands, millions of permutations of a junction with accompanying bezier path points 
Photoshop can also batch add noise to folders worth of images

**Press \[START CAPTURE\] which will download a new permutation 3 times pe second. _Remember to press \[STOP CAPTURE\]!_ When you do, a json file will also be downloaded with the intersection coordinate as well as bezier coordinates and matching file name for the jpg it refers to.**

↓ (You can also see the basic coordinates over a 0,0 central coordinate here:)

[https://openprocessing.org/sketch/1565698]


# ↓ More fake letter parts from programs I made before

With a bit more time than I have tonight, fake letters like these that I made a LONG time ago can be generated with intersection and path data. (hundreds of) thousands of letters can be downloaded quite quickly, you'll notice that number keys can change the thickness of letters when pressed repeatedly so variations of fonts also somewhat possible although real letters have greater stylistic variety than just these few styles...

[https://openprocessing.org/sketch/1003332]

[https://openprocessing.org/sketch/1002880]

[https://openprocessing.org/sketch/1003160]

[https://openprocessing.org/sketch/1566845]

[https://openprocessing.org/sketch/1614662]

# Chatgpt's ideas about where ML/DL might help...

# ML/DL next steps

## Where we are

- **No ML/DL used yet.** Everything is deterministic, pixel-only (Otsu, connected components, skeletonise/thinning, medial-axis “Voronoi” distance field, contouring, RDP, Chaikin).
    
- **Skeletons:** we use both `skeletonize` (thinning) and `medial_axis(..., return_distance=True)` (Voronoi/EDT). The latter is where some of the performance pain comes from.
    

## Can ML/DL improve things? Yes — in targeted places

Think “assist” modules, not a full rewrite. The biggest wins:

1. **Binarisation / clean-up (preprocess)**
    
    - A tiny U-Net (or even a classical random-forest on local features) can learn “ink probability” → fewer speckles, better edges than Otsu on tricky scans.
        
    - Drop-in replacement for our `binarise(gray)` step.
        
2. **Junction detection (Y3 & Y4)**
    
    - Train a **patch classifier** on 32×3232×3232×32 crops centred on skeleton pixels to label **junction vs curve** (+ junction type: T, Y, X) and an estimated **junction radius/orientation**.
        
    - This replaces our hand-tuned degree+cut tests and should reduce false hits on S-curves while giving better radii for masking.
        
3. **Centreline estimation (N4/N5)**
    
    - A light CNN that outputs a **centreline heatmap** (and optional width map). Soft-argmax + thinning on that heatmap yields a smoother, less wiggly centreline than plain skeletonise.
        
    - Helps the “altitude/Voronoi” path where normals from noisy skeletons create wobble.
        
4. **Corner snapping & curve bias (N2/Y5)**
    
    - A tiny model that classifies each contour vertex neighbourhood as **near-orthogonal / near-circular / general curve** and predicts a **snap direction**. Use it to drive smarter orthogonalisation and smoothing → cleaner icons and boxy glyphs.
        
5. **Routing thresholds (N5 vs Y5, should-be-path vs icon)**
    
    - Keep our hand-crafted features (LoA, IPQ, persistent junctions) but learn the decision rule with a **shallow tree or logistic regression**. This gives you tunable boundaries without hand-tweaking constants.
        
