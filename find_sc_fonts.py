
import os
import matplotlib.font_manager as font_manager

def scan_fonts():
    # Specific keywords for Simplified Chinese
    sc_keywords = [
        "sc", # Simplified Chinese (e.g. PingFang SC)
        "simplified",
        "simhei", 
        "simsun", 
        "songti sc",
        "heiti sc",
        "noto sans cjk sc",
        "source han sans sc"
    ]
    
    # Specific keywords for Traditional Chinese (to avoid)
    tc_keywords = [
        "tc", # Traditional Chinese
        "traditional",
        "mingliu",
        "pmingliu",
        "heiti tc",
        "songti tc"
    ]

    print("Scanning for Simplified Chinese fonts...")
    
    found_sc = []
    
    try:
        # Check standard system paths directly first as font_manager can be slow/incomplete
        system_paths = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts")
        ]
        
        for folder in system_paths:
            if not os.path.exists(folder): continue
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        path = os.path.join(root, file)
                        name = file.lower()
                        
                        # Check file name
                        is_sc = any(k in name for k in sc_keywords)
                        is_tc = any(k in name for k in tc_keywords)
                        
                        if is_sc and not is_tc:
                            found_sc.append(path)
                            
        # Also check font manager for installed names
        for f in font_manager.findSystemFonts():
            name = os.path.basename(f).lower()
            if any(k in name for k in sc_keywords) and not any(k in name for k in tc_keywords):
                found_sc.append(f)
                
        # Deduplicate
        found_sc = list(set(found_sc))
        
        print(f"Found {len(found_sc)} potential Simplified Chinese fonts:")
        for f in found_sc:
            print(f"  - {f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    scan_fonts()
