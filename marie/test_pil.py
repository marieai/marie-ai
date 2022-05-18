from PIL import Image,ImageDraw,ImageFont

# sample text and font
unicode_text = u"Hello World!"
# font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
font = ImageFont.truetype("../assets/fonts/FreeSansOblique.otf", 28)


#   inflating: FreeSans.ttf            
#   inflating: FreeSansBold.otf        
#   inflating: FreeSansBold.ttf        
#   inflating: FreeSansBoldOblique.otf  
#   inflating: FreeSansBoldOblique.ttf  
#   inflating: FreeSansOblique.otf     
#   inflating: FreeSansOblique.ttf     
#   inflating: sharefonts.net.txt 


# get the line size
text_width, text_height = font.getsize(unicode_text)

# create a blank canvas with extra space between lines
canvas = Image.new('RGB', (text_width + 10, text_height + 10), "orange")

# draw the text onto the text canvas, and use blue as the text color
draw = ImageDraw.Draw(canvas)
draw.text((5,5), u'Hello World!', 'blue', font)

# save the blank canvas to a file
canvas.save("/tmp/unicode-text.png", "PNG")
canvas.show()