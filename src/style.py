def lab_theme():
    font = "Sans-serif"

    axisColor = "#000000"
    gridColor = "#E7E8F0"
    legendBackgroundColor = '#F7F8FA'

    # Colors
    main_palette = ["#304FFE", 
                    "#FFC400",
                    "#FF206E",
                    "#00E5FF",
                    "#1A237E",
                    "#6C6AFF",
                    "#FF9100",
                    "#00CC88",
                    "#651FFF",
                    "#795548",
                    
                   ]
    sequential_palette = ["#1A237E", "#164394", "#1164A9", "#0D84BF", "#09A4D4", "#04C5EA", "#00E5FF"]


    return {
            "config": {
                "view": {
                    "height": 400
                },
                "title": {
                    "fontSize": 18,
                    "font": font,
                    "anchor": "mid", # equivalent of left-aligned.
                    "fontColor": "#000000"
                },
                "axisX": {
                    "domain": True,
                    "domainColor": axisColor,
                    "domainWidth": 1,
                    "grid": False,
                    "labelFont": font,
                    "labelFontSize": 14,
                    "labelAngle": 0, 
                    "tickColor": axisColor,
                    "tickSize": 5, # default, including it just to show you can change it
                    "titleFont": font,
                    "titleFontSize": 14,
                    "titlePadding": 10, # guessing, not specified in styleguide
                    "title": "X Axis Title (units)", 
                },
                "axisY": {
                    "domain": False,
                    "grid": True,
                    "gridColor": gridColor,
                    "gridDash": [5],
                    "gridWidth": 1,
                    "labelFont": font,
                    "labelFontSize": 14,
                    "labelAngle": 0, 
                    "ticks": False, # even if you don't have a "domain" you need to turn these off.
                    "titleFont": font,
                    "titleFontSize": 14,
                    "titlePadding": 10, # guessing, not specified in styleguide
                    "title": "Y Axis Title (units)", 
                    # titles are by default vertical left of axis so we need to hack this 
                    "titleAngle": 0, # horizontal
                    "titleY": -10, # move it up
                    "titleX": 18, # move it to the right so it aligns with the labels 
                },
                "header": {  # used in row or column grouped chart
                    "labelAngle": 0,
                    "labelOrient": "top",
                    "labelFontSize": 14,
                    "labelLimit": 150,
                    "labelPadding": 4,
                },
                "range": {
                    "category": main_palette,
                    "diverging": sequential_palette,
                },
                "legend": {
                    "labelFont": font,
                    "labelFontSize": 14,
                    "symbolType": "square", # just 'cause
                    "symbolSize": 100, # default
                    "titleFont": font,
                    "titleFontSize": 14,
                    "strokeColor": legendBackgroundColor,
                    "fillColor": legendBackgroundColor,
                    "padding": 10,
                    "cornerRadius": 10,
                    "title": "", # set it to no-title by default
                    "orient": "top", # so it's right next to the y-axis
                    "offset": 10, # literally right next to the y-axis.
                },
                "view": {
                    "strokeDash": [5], # altair uses gridlines to box the area where the data is visualized. This takes that off.
                    #"stroke": gridColor
                },
                 ### MARKS CONFIGURATIONS ###
        
               "line": {
                   "strokeWidth": 3,
               },
               "point": {
                   "filled": True,
               },
               "text": {
                   "font": font,
                   "fontSize": 14,
                   "align": "right",
                   "fontWeight": 400,
                   "size": 14,
               }, 
               "bar": {
                #"size": 40,
                "binSpacing": 2,
                "continuousBandSize": 30,
                #"discreteBandSize": 30,
                "stroke": False,
            },
            }
        }
        