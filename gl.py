#Maria Jose Castro Lemus 
#181202
#Graficas por Computadora - 10
#Lab 1: SR1 Point

import struct 

def char(c):
    return struct.pack('=c', c.encode('ascii'))
def word(c):
    return struct.pack('=h', c)
def dword(c):
    return struct.pack('=l', c)
def color(r, g, b):
    return bytes([b, g, r]) 


class Render(object):
    def __init__(self):
        #self.width = width
        #self.height = height
        self.framebuffer =[]
        #self.clear()
        #self.glCreateWindow()

    def glInit(self):
        pass

    def clear(self, r, g,b):
        self.framebuffer= [
        [color(r,g,b) for x in range(self.width)]
        for y in range(self.height)
        ]

    def  glClear(self):
        self.clear()

    def glClearcolor(self, r, g, b):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        self.clear(r, g, b)

    def glColor(self, r,g,b):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        return color(r, g, b)
        
    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height
        #r = Render(width,height)

    def glViewport(self, x, y, width, height):
        self.viewPortWidth = width
        self.viewPortHeight = height
        self.xViewPort = x
        self.yViewPort = y

    def glVertex(self, x,y):
        calcX = int((x+1)*(self.viewPortWidth/2)+self.xViewPort)
        calcY = int((y+1)*(self.viewPortHeight/2)+self.yViewPort)
        self.point(calcX, calcY)


    def write(self, filename):
        f = open(filename, 'bw')
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        #image header 
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        #pixel data
        for x in range(self.width):
            for y in range(self.height):
                    #print(self.width,' ', self.height,' hola')
                    f.write(self.framebuffer[y][x])

        f.close()

    #function dot
    def point(self, x, y):
        self.framebuffer[x][y] = self.glColor(0.137254902,0.7607843137,0.3490196078)    

        #Referencia del repositorio ejemplo de dennis
    def glFinish(self, filename='out.bmp'):
        self.write(filename)
        try:
          from wand.image import Image
          from wand.display import display

          with Image(filename=filename) as image:
            display(image)
        except ImportError:
          pass  # do nothing if no wand is installed




bitmap = Render()

bitmap.glCreateWindow(100,200)
bitmap.glClearcolor(0.3647058824, 0.137254902, 0.7607843137)
bitmap.glViewport(5, 5, 75, 75)
bitmap.glVertex( -1, -1)
bitmap.glFinish()
