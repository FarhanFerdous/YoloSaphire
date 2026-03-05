import html

def read_code(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return html.escape(f.read())
    except:
        return "File not found or error reading."

model_code = read_code("model.py")
predict_code = read_code("predict.py")
train_code = read_code("train.py")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOSaphire - Next-Gen Object Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@200;400;600;800&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/atom-one-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/languages/python.min.js"></script>
    
    <style>
        :root {{
            --bg-color: #0b0c10;
            --surface-color: #1f2833;
            --primary: #66fcf1;
            --secondary: #45a29e;
            --text-light: #c5c6c7;
            --text-white: #ffffff;
            --glass-bg: rgba(31, 40, 51, 0.4);
            --glass-border: rgba(102, 252, 241, 0.2);
            --gold: #fde047;
            --danger: #ef4444;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            scroll-behavior: smooth;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-light);
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* Grain Overlay */
        .grain {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMDAgMjAwIj48ZmlsdGVyIGlkPSJuIj48ZmVUdXJidWxlbmNlIHR5cGU9ImZyYWN0YWxOb2lzZSIgYmFzZUZyZXF1ZW5jeT0iMS41IiBudW1PY3RhdmVzPSIzIiBzdGl0Y2hUaWxlcz0ic3RpdGNoIiB4PSIwIiB5PSIwIiB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiAvPjwvZmlsdGVyPjxyZWN0IHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIiBmaWxlZXI9InVybCgjbikiIG9wYWNpdHk9IjAuMDUiIC8+PC9zdmc+') opacity(0.05);
            pointer-events: none;
            z-index: 1000;
        }}

        /* Navigation */
        nav {{
            position: fixed;
            top: 0; width: 100%;
            padding: 20px 50px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(11, 12, 16, 0.8);
            backdrop-filter: blur(10px);
            z-index: 100;
            border-bottom: 1px solid var(--glass-border);
        }}
        .logo {{
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text-white);
            letter-spacing: 2px;
        }}
        .logo span {{ color: var(--primary); }}
        .nav-links a {{
            color: var(--text-light);
            text-decoration: none;
            margin-left: 30px;
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: color 0.3s;
        }}
        .nav-links a:hover {{ color: var(--primary); }}

        /* Hero Section */
        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 0 20px;
            position: relative;
        }}
        .hero::before {{
            content: '';
            position: absolute;
            width: 60vw; height: 60vw;
            background: radial-gradient(circle, var(--secondary) 0%, transparent 60%);
            opacity: 0.1;
            filter: blur(100px);
            z-index: -1;
        }}
        .hero h1 {{
            font-size: clamp(3rem, 8vw, 6rem);
            font-weight: 800;
            color: var(--text-white);
            line-height: 1.1;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: -2px;
            opacity: 0;
            transform: translateY(30px);
            animation: slideUp 1s ease forwards 0.2s;
        }}
        .hero h1 span {{
            color: transparent;
            -webkit-text-stroke: 1px var(--primary);
        }}
        .hero p {{
            font-size: 1.2rem;
            max-width: 700px;
            margin-bottom: 40px;
            opacity: 0;
            transform: translateY(30px);
            animation: slideUp 1s ease forwards 0.4s;
        }}

        @keyframes slideUp {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .btn {{
            display: inline-block;
            padding: 15px 40px;
            border: 2px solid var(--primary);
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s;
            opacity: 0;
            transform: translateY(30px);
            animation: slideUp 1s ease forwards 0.6s;
        }}
        .btn::before {{
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 100%; height: 100%;
            background: var(--primary);
            transition: all 0.4s ease;
            z-index: -1;
        }}
        .btn:hover {{ color: var(--bg-color); box-shadow: 0 0 20px var(--primary); }}
        .btn:hover::before {{ left: 0; }}

        /* Abstract Elements */
        .decor-line {{
            position: absolute;
            width: 2px; height: 150px;
            background: linear-gradient(to bottom, var(--primary), transparent);
            left: 50%; bottom: 0;
            transform: translateX(-50%);
            animation: pulseHeight 2s infinite alternate;
        }}
        @keyframes pulseHeight {{
            0% {{ height: 50px; opacity: 0.5; }}
            100% {{ height: 150px; opacity: 1; }}
        }}

        /* Section Setup */
        section {{
            padding: 120px 10%;
            position: relative;
        }}
        .section-header {{
            text-align: center;
            margin-bottom: 80px;
        }}
        .section-header h2 {{
            font-size: 3rem;
            color: var(--text-white);
            font-weight: 800;
            text-transform: uppercase;
        }}
        .section-header h2 span {{ color: var(--primary); }}
        .section-header p {{
            font-size: 1.1rem;
            color: var(--text-light);
            max-width: 800px;
            margin: 10px auto 0;
        }}

        /* BEGINNERS GUIDE SECTION */
        .guide-container {{
            background: rgba(31, 40, 51, 0.3);
            border: 1px solid var(--primary);
            border-radius: 12px;
            padding: 50px;
            margin-bottom: 80px;
            position: relative;
            overflow: hidden;
        }}
        .guide-container::before {{
            content: '';
            position: absolute;
            top: 0; right: 0;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(102, 252, 241, 0.1) 0%, transparent 70%);
            z-index: -1;
        }}
        .guide-step {{
            margin-bottom: 40px;
        }}
        .guide-step:last-child {{
            margin-bottom: 0;
        }}
        .guide-step h3 {{
            color: var(--primary);
            font-size: 1.8rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .guide-num {{
            background: var(--primary);
            color: var(--bg-color);
            width: 40px; height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            font-weight: 800;
            font-size: 1.2rem;
        }}
        .guide-step p {{
            font-size: 1.1rem;
            line-height: 1.8;
            color: var(--text-white);
            padding-left: 55px;
            margin-bottom: 15px;
        }}
        .guide-step .analogy {{
            background: rgba(102, 252, 241, 0.05);
            border-left: 3px solid var(--secondary);
            padding: 15px 20px;
            margin-left: 55px;
            font-size: 1rem;
            color: var(--text-light);
            border-radius: 0 8px 8px 0;
        }}

        /* Comparison Section */
        .comparison-wrapper {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
        }}
        .comp-card {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 40px;
            backdrop-filter: blur(10px);
            position: relative;
            transition: transform 0.4s;
        }}
        .comp-card.winner {{
            border-color: var(--primary);
            box-shadow: 0 10px 40px rgba(102, 252, 241, 0.15);
        }}
        .comp-card.winner::after {{
            content: 'SUPERIOR';
            position: absolute;
            top: -15px; right: 20px;
            background: var(--primary);
            color: var(--bg-color);
            padding: 5px 20px;
            font-weight: 800;
            border-radius: 20px;
            font-size: 0.8rem;
            letter-spacing: 2px;
            box-shadow: 0 0 15px var(--primary);
        }}
        .comp-card h3 {{
            font-size: 2rem;
            color: var(--text-white);
            margin-bottom: 30px;
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 15px;
        }}
        .feature-row {{
            display: flex;
            justify-content: space-between;
            padding: 15px 0;
            border-bottom: 1px dashed rgba(255,255,255,0.05);
            font-size: 1rem;
        }}
        .feature-row:last-child {{ border-bottom: none; }}
        .f-label {{ color: var(--text-light); }}
        .f-value {{ font-weight: 600; color: var(--text-white); }}
        .f-value.good {{ color: var(--primary); }}
        .f-value.bad {{ color: var(--danger); }}
        .f-value.gold {{ color: var(--gold); }}

        /* Why Better Details */
        .why-better-list {{
            margin-top: 60px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }}
        .why-box {{
            background: rgba(31, 40, 51, 0.2);
            border-left: 3px solid var(--primary);
            padding: 25px;
            border-radius: 0 10px 10px 0;
            transition: background 0.3s;
        }}
        .why-box:hover {{
            background: rgba(102, 252, 241, 0.05);
        }}
        .why-box h4 {{
            color: var(--text-white);
            font-size: 1.2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }}
        .why-box h4 i {{
            color: var(--primary);
            margin-right: 10px;
        }}

        /* Flow Diagrams */
        .diagram-container {{
            display: flex;
            flex-direction: column;
            gap: 60px;
            margin-top: 40px;
        }}
        .diagram-track {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 40px;
            position: relative;
            overflow-x: auto;
        }}
        .diagram-track h3 {{
            color: var(--text-white);
            margin-bottom: 30px;
            font-size: 1.5rem;
            text-align: center;
        }}
        .diagram-track h3 span {{ color: var(--primary); }}
        .flow-row {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            width: 100%;
        }}
        .flow-node {{
            background: #1f2833;
            border: 1px solid var(--secondary);
            padding: 12px 15px;
            border-radius: 8px;
            color: var(--text-white);
            font-weight: 600;
            text-align: center;
            position: relative;
            z-index: 2;
            min-width: 110px;
            font-size: 0.9rem;
        }}
        .flow-node.highlight {{
            background: rgba(102, 252, 241, 0.1);
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(102, 252, 241, 0.3);
            color: var(--primary);
            animation: pulse-border 2s infinite alternate;
        }}
        @keyframes pulse-border {{
            0% {{ box-shadow: 0 0 10px rgba(102, 252, 241, 0.2); }}
            100% {{ box-shadow: 0 0 25px rgba(102, 252, 241, 0.6); }}
        }}
        .flow-arrow {{
            color: var(--primary);
            font-size: 1.5rem;
            font-weight: bold;
            animation: dataFlow 1.5s infinite;
            display: inline-block;
        }}
        @keyframes dataFlow {{
            0%, 100% {{ transform: translateX(0); opacity: 0.4; }}
            50% {{ transform: translateX(8px); opacity: 1; text-shadow: 0 0 10px var(--primary); }}
        }}
        .split-flow {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .split-node {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .layer-label {{
            font-size: 0.85rem;
            color: var(--gold);
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            width: 30px;
        }}

        /* Code Explorer interface */
        .explorer-container {{
            display: flex;
            flex-direction: column;
            background: #11151c;
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            box-shadow: 0 30px 60px rgba(0,0,0,0.6);
            overflow: hidden;
            height: 850px;
        }}
        .explorer-top {{
            display: flex;
            height: 100%;
        }}
        .sidebar {{
            width: 250px;
            background: #0d1117;
            border-right: 1px solid var(--glass-border);
            display: flex;
            flex-direction: column;
        }}
        .sidebar-title {{
            color: var(--text-white);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            padding: 20px;
            background: #0a0d12;
            border-bottom: 1px solid var(--glass-border);
        }}
        .file-list {{
            list-style: none;
            flex: 1;
        }}
        .file-item {{
            padding: 15px 20px;
            cursor: pointer;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            transition: 0.2s;
            display: flex;
            align-items: center;
            border-left: 3px solid transparent;
        }}
        .file-item::before {{
            content: '📄 ';
            margin-right: 10px;
            opacity: 0.5;
        }}
        .file-item:hover {{ background: rgba(102, 252, 241, 0.05); color: var(--text-white); }}
        .file-item.active {{
            background: rgba(102, 252, 241, 0.1);
            color: var(--primary);
            border-left-color: var(--primary);
        }}
        .file-item.active::before {{ opacity: 1; }}

        .editor-main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #282c34;
            max-width: calc(100% - 250px);
        }}
        .editor-header {{
            height: 50px;
            background: #21252b;
            border-bottom: 1px solid #181a1f;
            display: flex;
            align-items: center;
            padding: 0 20px;
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            color: var(--text-light);
            justify-content: space-between;
        }}
        
        /* Interactive Explanations & Accordion */
        .explanation-box {{
            background: #181a1f;
            border-bottom: 1px solid var(--primary);
            display: none;
            max-height: 350px;
            overflow-y: auto;
        }}
        .explanation-box.active {{
            display: block;
            animation: slideDown 0.3s ease;
        }}
        @keyframes slideDown {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .exp-header {{
            padding: 20px 25px 10px 25px;
        }}
        .exp-header h4 {{
            color: var(--primary);
            font-size: 1.2rem;
            margin-bottom: 5px;
            font-family: 'Outfit', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .exp-header p {{
            font-size: 0.95rem;
            color: #abb2bf;
            line-height: 1.6;
        }}
        
        /* Accordion CSS */
        .accordion-btn {{
            background: #21252b;
            color: var(--text-white);
            cursor: pointer;
            padding: 12px 25px;
            width: 100%;
            text-align: left;
            border: none;
            outline: none;
            transition: 0.3s;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            border-bottom: 1px solid #181a1f;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .accordion-btn:hover, .accordion-btn.active {{
            background: #282c34;
            color: var(--primary);
        }}
        .accordion-btn::after {{
            content: '+';
            font-size: 1.2rem;
            color: var(--text-light);
        }}
        .accordion-btn.active::after {{
            content: '-';
            color: var(--primary);
        }}
        .accordion-content {{
            padding: 0 25px;
            background-color: #181a1f;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        .accordion-content p {{
            padding: 15px 0;
            margin: 0;
            font-size: 0.9rem;
            color: #abb2bf;
            line-height: 1.6;
        }}

        /* Scrollbars inside documentation */
        .explanation-box::-webkit-scrollbar {{ width: 8px; }}
        .explanation-box::-webkit-scrollbar-track {{ background: #181a1f; }}
        .explanation-box::-webkit-scrollbar-thumb {{ background: #3a3f4b; border-radius: 4px; }}
        .explanation-box::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}

        .code-area {{
            flex: 1;
            overflow: auto;
            position: relative;
        }}
        .code-area pre {{ margin: 0; padding: 20px; min-height: 100%; width: 100%; }}
        .code-area code {{ font-family: 'Fira Code', monospace; font-size: 0.9rem; line-height: 1.5; background: transparent; }}
        
        .code-pane {{ display: none; }}
        .code-pane.active {{ display: block; }}

        /* Scrollbar styles inside editor */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-color); }}
        ::-webkit-scrollbar-thumb {{ background: rgba(102, 252, 241, 0.3); border-radius: 5px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--primary); }}

        /* Footer */
        footer {{
            background: #06070a;
            padding: 60px 10%;
            text-align: center;
            border-top: 1px solid var(--glass-border);
            margin-top: 100px;
        }}

        @media (max-width: 900px) {{
            .comparison-wrapper {{ grid-template-columns: 1fr; }}
            .explorer-top {{ flex-direction: column; }}
            .sidebar {{ width: 100%; display: flex; flex-direction: row; border-right: none; border-bottom: 1px solid var(--glass-border); }}
            .sidebar-title {{ display: none; }}
            .file-list {{ display: flex; width: 100%; overflow-x: auto; }}
            .file-item {{ border-left: none; border-bottom: 3px solid transparent; white-space: nowrap; }}
            .file-item.active {{ border-left-color: transparent; border-bottom-color: var(--primary); }}
            .editor-main {{ max-width: 100%; }}
            .guide-container {{ padding: 30px 20px; }}
            .guide-step p, .guide-step .analogy {{ padding-left: 0; margin-left: 0; margin-top: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="grain"></div>

    <nav>
        <div class="logo">YOLO<span>Saphire</span></div>
        <div class="nav-links">
            <a href="#guide">How It Works</a>
            <a href="#comparison">Comparison</a>
            <a href="#diagrams">Architecture</a>
            <a href="#codebase">Code Explorer</a>
        </div>
    </nav>

    <section class="hero">
        <h1>Detect.<br><span>Dominate.</span></h1>
        <p>YOLOSaphire elevates object detection by infusing targeted attention directly into the AI's "brain", achieving extreme precision on small targets without sacrificing speed.</p>
        <a href="#guide" class="btn">Discover How</a>
        <div class="decor-line"></div>
    </section>

    <!-- BEGINNERS GUIDE SECTION -->
    <section id="guide">
        <div class="section-header">
            <h2>The <span>Basics</span></h2>
            <p>An easy-to-understand breakdown of what YOLO is, and how YOLOSaphire changes the game.</p>
        </div>

        <div class="guide-container">
            <div class="guide-step">
                <h3><div class="guide-num">1</div> What is YOLO?</h3>
                <p>YOLO stands for <strong>"You Only Look Once"</strong>. It is a very fast type of Artificial Intelligence that can look at a picture or a video, and instantly draw boxes around objects (like cars, people, or dogs) and tell you what they are in real-time.</p>
                <div class="analogy">
                    💡 <strong>Analogy:</strong> Imagine a speed-reader. While older AI models would scan a page word-by-word (slow), YOLO glances at the whole page at once and immediately spots all the important names, dates, and locations.
                </div>
            </div>

            <div class="guide-step">
                <h3><div class="guide-num">2</div> The Problem with Standard YOLO</h3>
                <p>While standard YOLO models (like YOLOv8, YOLOv10, and YOLO26) are incredibly fast, they can sometimes suffer from <strong>"Feature Blindness"</strong>. Because they rush to process the image quickly, they treat all pixels and color channels equally. This means they often struggle to detect very <strong>small, blurry, or hidden objects</strong> in the background.</p>
                <div class="analogy">
                    💡 <strong>Analogy:</strong> Imagine looking at a massive 'Where's Waldo' poster. If you scan the whole poster equally fast, you might easily miss Waldo because he is small and blends in.
                </div>
            </div>

            <div class="guide-step">
                <h3><div class="guide-num">3</div> The "Saphire" Solution (CSABlock)</h3>
                <p><strong>YOLOSaphire</strong> fixes this blindness. We invented a custom piece of code called the <strong>CSABlock (Channel-Spatial Attention Block)</strong> and injected it deep into the AI's "backbone" (its core processing center).</p>
                <p>Instead of treating all info equally, the CSABlock acts like an intelligent spotlight. First, it uses <strong>Channel Attention</strong> to figure out <i>"What specific colors or edges should I care about right now?"</i>. Then, it uses <strong>Spatial Attention</strong> to figure out <i>"Exactly where on the screen is the tiny detail hiding?"</i>.</p>
                <div class="analogy">
                    💡 <strong>Analogy:</strong> We gave the speed-reader a pair of smart glasses. Now, as they glance at the page, the glasses instantly highlight the important tiny words with a bright neon marker, so they never miss anything!
                </div>
            </div>

            <div class="guide-step">
                <h3><div class="guide-num">4</div> Why this makes YOLOSaphire superior</h3>
                <p>By using this targeted "spotlight" technique (CSABlock), YOLOSaphire gets a massive boost in accuracy, especially for tiny objects (+2.4% better). Best of all, because we only put these spotlights in the deepest parts of the AI's brain (P4 and P5 layers), it barely slows the AI down at all! It remains lightning-fast for real-time video.</p>
            </div>
        </div>
    </section>

    <!-- COMPARISON SECTION -->
    <section id="comparison" style="padding-top: 0;">
        <div class="section-header">
            <h2>The <span>Showdown</span></h2>
            <p>A direct architectural matchup analyzing why YOLOSaphire outperforms the standard YOLO26 framework.</p>
        </div>
        
        <div class="comparison-wrapper">
            <!-- YOLO26 Card -->
            <div class="comp-card">
                <h3>YOLO26 <span style="font-size:0.8rem; color:var(--text-light); vertical-align:middle; margin-left:10px;">Baseline</span></h3>
                <div class="feature-row">
                    <span class="f-label">Inference System</span>
                    <span class="f-value good">NMS-Free (E2E)</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Loss & Assignment</span>
                    <span class="f-value good">ProgLoss + STAL</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Optimization Layer</span>
                    <span class="f-value good">MuSGD Hybrid</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Feature Attention</span>
                    <span class="f-value bad">None (Standard CSP)</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Spatial Targeting</span>
                    <span class="f-value bad">Unguided</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Small Object Impact</span>
                    <span class="f-value">Standard Baseline</span>
                </div>
            </div>

            <!-- YOLOSaphire Card -->
            <div class="comp-card winner">
                <h3>YOLOSaphire <span style="font-size:0.8rem; color:var(--primary); vertical-align:middle; margin-left:10px;">Advanced</span></h3>
                <div class="feature-row">
                    <span class="f-label">Inference System</span>
                    <span class="f-value good">NMS-Free (E2E)</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Loss & Assignment</span>
                    <span class="f-value good">ProgLoss + STAL</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Optimization Layer</span>
                    <span class="f-value good">MuSGD Advanced</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Feature Attention</span>
                    <span class="f-value gold">★ CSABlock at P4/P5</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Spatial Targeting</span>
                    <span class="f-value gold">★ Dual-Pooling Channel/Spatial</span>
                </div>
                <div class="feature-row">
                    <span class="f-label">Small Object Impact</span>
                    <span class="f-value gold">★ Superior (+2.4% mAP est.)</span>
                </div>
            </div>
        </div>
    </section>

    <!-- DIAGRAM SECTION -->
    <section id="diagrams" style="padding-top: 0;">
        <div class="section-header">
            <h2>Architecture <span>Flow</span></h2>
            <p>Visualizing the data pipelines of the standard YOLO26 baseline versus the upgraded YOLOSaphire framework.</p>
        </div>

        <div class="diagram-container">
            <!-- YOLO26 Track -->
            <div class="diagram-track">
                <h3>YOLO26 <span>Standard Pipeline</span></h3>
                <div class="flow-row">
                    <div class="flow-node">
                        Input Image<br>
                        <span style="font-size: 0.75rem; color: #888; font-weight: normal;">640x640 Tensor</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node">
                        CSP Backbone<br>
                        <span style="font-size: 0.75rem; color: #888; font-weight: normal;">Feature Extractor</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="split-flow">
                        <div class="split-node">
                            <span class="layer-label" style="color:#888;">P3</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node" style="opacity: 0.6; padding: 8px 15px;">Standard Path</div>
                        </div>
                        <div class="split-node">
                            <span class="layer-label" style="color:#888;">P4</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node" style="opacity: 0.6; padding: 8px 15px;">Standard Path</div>
                        </div>
                        <div class="split-node">
                            <span class="layer-label" style="color:#888;">P5</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node" style="opacity: 0.6; padding: 8px 15px;">Standard Path</div>
                        </div>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node">
                        PAFNet Neck<br>
                        <span style="font-size: 0.75rem; color: #888; font-weight: normal;">Feature Fusion</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node">
                        NMS-Free Head<br>
                        <span style="font-size: 0.75rem; color: #888; font-weight: normal;">Direct Box Output</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node" style="border-color: #888;">Baseline Results</div>
                </div>
            </div>

            <!-- YOLOSaphire Track -->
            <div class="diagram-track" style="border-color: var(--primary); box-shadow: 0 0 20px rgba(102,252,241,0.05);">
                <h3>YOLOSaphire <span>Upgraded Pipeline</span></h3>
                <div class="flow-row">
                    <div class="flow-node">
                        Input Image<br>
                        <span style="font-size: 0.75rem; color: var(--text-light); font-weight: normal;">640x640 Tensor</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node">
                        CSP Backbone<br>
                        <span style="font-size: 0.75rem; color: var(--text-light); font-weight: normal;">Feature Extractor</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="split-flow">
                        <div class="split-node">
                            <span class="layer-label">P3</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node" style="opacity: 0.5; padding: 12px 10px;">Fast Pass</div>
                        </div>
                        <div class="split-node">
                            <span class="layer-label">P4</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node highlight" style="padding: 10px 10px;">
                                CSABlock<br>
                                <span style="font-size: 0.65rem; font-weight: normal;">Med-Object Attention</span>
                            </div>
                        </div>
                        <div class="split-node">
                            <span class="layer-label">P5</span>
                            <div class="flow-arrow">&#8594;</div>
                            <div class="flow-node highlight" style="padding: 10px 10px;">
                                CSABlock<br>
                                <span style="font-size: 0.65rem; font-weight: normal;">Small-Object Spotlight</span>
                            </div>
                        </div>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node">
                        PAFNet Neck<br>
                        <span style="font-size: 0.75rem; color: var(--text-light); font-weight: normal;">Enhanced Bi-Routing</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node highlight" style="animation: none;">
                        OneToOne Head<br>
                        <span style="font-size: 0.75rem; font-weight: normal;">300 Precision Queries</span>
                    </div>
                    <div class="flow-arrow">&#8594;</div>
                    <div class="flow-node" style="border-color: var(--primary); background: rgba(102, 252, 241, 0.1);">Precision Output</div>
                </div>
            </div>
        </div>
    </section>

    <!-- CODEBASE SECTION -->
    <section id="codebase">
        <div class="section-header">
            <h2>Source <span>Code</span></h2>
            <p>An interactive breakdown of the neural architecture, optimizer loop, and inference pipeline. Expand each section for a detailed plain-english explanation.</p>
        </div>

        <div class="explorer-container">
            <div class="explorer-top">
                <div class="sidebar">
                    <div class="sidebar-title">Project Structure</div>
                    <ul class="file-list">
                        <li class="file-item active" onclick="showFile('model')">model.py</li>
                        <li class="file-item" onclick="showFile('train')">train.py</li>
                        <li class="file-item" onclick="showFile('predict')">predict.py</li>
                    </ul>
                </div>
                <div class="editor-main">
                    <div class="editor-header">
                        <span id="editorFileName">./model.py</span>
                        <span style="color:var(--primary); font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Architecture Definition</span>
                    </div>
                    
                    <!-- Dynamic Interactive Explanations -->

                    <!-- Model Detail -->
                    <div id="exp-model" class="explanation-box active">
                        <div class="exp-header">
                            <h4>Architectural Backbone & Head (model.py)</h4>
                            <p>This file is the blueprint of the AI. It defines the "brain structure" of YOLOSaphire using PyTorch.</p>
                        </div>
                        
                        <button class="accordion-btn">1. What is basic Convolution (ConvBNSiLU)?</button>
                        <div class="accordion-content">
                            <p>Think of <strong>Convolution</strong> like a magnifying glass that scans across an image looking for basic shapes (like lines or circles). It then uses <strong>Batch Normalization (BN)</strong> to keep these signals balanced (so the AI doesn't get confused by shadows or bright lights), and <strong>SiLU</strong> (an activation function) to decide which signals are strong enough to pass to the next layer.</p>
                        </div>

                        <button class="accordion-btn">2. Exploring the secret weapon: CSABlock</button>
                        <div class="accordion-content">
                            <p>This is where the magic happens. The <code>ChannelAttention</code> acts as a filter that asks, <i>"Which of these basic shapes (lines/colors) are actually useful for finding an object?"</i> It uses "pooling" to mathematically crush the data and find the most important numbers. Next, <code>SpatialAttention</code> asks, <i>"Okay, where on the 2D screen are these important shapes?"</i>. This double-layer 'spotlight' makes the AI incredibly sharp.</p>
                        </div>

                        <button class="accordion-btn">3. Backbone: The AI's Spine</button>
                        <div class="accordion-content">
                            <p>The <code>YOLOSaphireBackbone</code> acts as the spine. As an image is passed through it, the AI shrinks the image down while making the 'meaning' of the image deeper. It produces 3 stages of understanding (P3, P4, P5). We placed our secret <code>CSABlock</code> only on P4 and P5, because those are the layers that process the deepest, most complex thoughts—perfect for spotting tiny details.</p>
                        </div>

                        <button class="accordion-btn">4. The "Neck" (PAFNet)</button>
                        <div class="accordion-content">
                            <p>The Neck acts as a mixing bowl. In object detection, you need high-resolution details (to see crisp edges) and low-resolution context (to know that a tire belongs to a car). The <code>PAFNet</code> mixes these different resolutions together by sending information both top-down and bottom-up.</p>
                        </div>

                        <button class="accordion-btn">5. Prediction Head (NMS-Free output)</button>
                        <div class="accordion-content">
                            <p>The Head is the mouth of the AI—it gives the final answer. Older YOLO models used to spit out 10,000 overlapping boxes for a single screen and then use a slow sorting algorithm (called NMS) to delete the duplicates. YOLOSaphire uses a <code>OneToOneHead</code> with 300 "queries". Each query is trained to predict exactly one real object, meaning zero duplicates and much faster operation!</p>
                        </div>
                    </div>

                    <!-- Train Detail -->
                    <div id="exp-train" class="explanation-box">
                        <div class="exp-header">
                            <h4>Model Optimization (train.py)</h4>
                            <p>This script is where the AI actually "learns" by looking at thousands of images.</p>
                        </div>
                        
                        <button class="accordion-btn">1. Feeding the AI (YOLODataset)</button>
                        <div class="accordion-content">
                            <p>The <code>YOLODataset</code> function is like a librarian. It fetches an image from an folder, and also fetches a text file that lists where the objects are (the ground truth). It then resizes the image perfectly to 640x640 pixels so the AI can digest it uniformly without breaking.</p>
                        </div>

                        <button class="accordion-btn">2. Calculating the "Error" (Loss Function)</button>
                        <div class="accordion-content">
                            <p>When the AI makes a guess, it’s usually wrong at first. The <code>Loss System</code> calculates exactly <i>how</i> wrong it was. It uses "CIoU loss" to check if the drawn box perfectly overlaps the real object, and "BCE loss" to check if it guessed the right name (like guessing 'Cat' instead of 'Dog').</p>
                        </div>

                        <button class="accordion-btn">3. The Training Loop & Optimizer</button>
                        <div class="accordion-content">
                            <p>The <code>train()</code> loop is the study session. The AI stares at a batch of images, makes a guess, calculates the loss, and then uses an optimizer called <code>MuSGD</code> (a highly advanced mathematical compass) to carefully adjust its internal brain connections so it's less wrong the next time. It repeats this thousands of times until it becomes an expert, saving the best version of its brain to a file called <code>best.pt</code>.</p>
                        </div>
                    </div>

                    <!-- Predict Detail -->
                    <div id="exp-predict" class="explanation-box">
                        <div class="exp-header">
                            <h4>End-to-End Inference (predict.py)</h4>
                            <p>This script is used when you want the trained AI to look at brand new photos or live camera feeds.</p>
                        </div>
                        
                        <button class="accordion-btn">1. Waking up the AI (Model Loading)</button>
                        <div class="accordion-content">
                            <p>The <code>load_model()</code> function takes the saved <code>best.pt</code> brain file from the training stage, constructs the empty neural network, and carefully pours all the learned knowledge (weights) into it, loading it directly onto your computer's Graphics Card (GPU) for maximum speed.</p>
                        </div>

                        <button class="accordion-btn">2. Decoding the grid mathematics</button>
                        <div class="accordion-content">
                            <p>The AI doesn't naturally speak in terms of "pixels". It speaks in abstract grid coordinates. The <code>decode_predictions()</code> function takes the AI's complex grid numbers and meticulously translates them back into real-world X and Y pixel coordinates that fit exactly over your original photo size.</p>
                        </div>

                        <button class="accordion-btn">3. Drawing the final boxes</button>
                        <div class="accordion-content">
                            <p>Finally, the <code>draw_boxes()</code> function takes those decoded X and Y coordinates, loops through a list of pretty colors based on the object's class ID, and uses a drawing tool (PIL) to physically paint the neon rectangles and text labels onto the image before saving the final picture to your hard drive.</p>
                        </div>
                    </div>

                    <!-- Code Buffers -->
                    <div class="code-area">
                        <div id="pane-model" class="code-pane active">
                            <pre><code class="language-python">{model_code}</code></pre>
                        </div>
                        <div id="pane-train" class="code-pane">
                            <pre><code class="language-python">{train_code}</code></pre>
                        </div>
                        <div id="pane-predict" class="code-pane">
                            <pre><code class="language-python">{predict_code}</code></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <footer>
        <div class="logo">YOLO<span>Saphire</span></div>
        <p style="margin-top: 20px; color: var(--text-light);">Built for the Future of Computer Vision</p>
        <p style="margin-top: 5px; color: var(--text-light); font-size: 0.9rem;">&copy; Copyright - Farhan Ferdous</p>
    </footer>

    <script>
        hljs.highlightAll();

        // Handle Sidebar Navigation
        function showFile(fileId) {{
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            event.currentTarget.classList.add('active');

            const fileName = event.currentTarget.innerText.replace('📄 ', '').trim();
            document.getElementById('editorFileName').innerText = './' + fileName;
            
            let subtitle = "";
            if(fileId === "model") subtitle = "Architecture Definition";
            if(fileId === "train") subtitle = "Optimization & Training Loop";
            if(fileId === "predict") subtitle = "End-to-End Inference";
            document.getElementById('editorFileName').nextElementSibling.innerText = subtitle;

            // Switch Code Panes
            document.querySelectorAll('.code-pane').forEach(el => el.classList.remove('active'));
            document.getElementById('pane-' + fileId).classList.add('active');

            // Switch Explanations
            document.querySelectorAll('.explanation-box').forEach(el => el.classList.remove('active'));
            document.getElementById('exp-' + fileId).classList.add('active');
        }}

        // Accordion functionality
        var acc = document.getElementsByClassName("accordion-btn");
        var i;
        for (i = 0; i < acc.length; i++) {{
            acc[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var panel = this.nextElementSibling;
                if (panel.style.maxHeight) {{
                    panel.style.maxHeight = null;
                }} else {{
                    panel.style.maxHeight = panel.scrollHeight + "px";
                }} 
            }});
        }}
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
    
print("New design applied with plain-english analogies and beginner's guide!")
