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
        calcX = round((x+1)*(self.viewPortWidth/2)+self.xViewPort)
        calcY = round((y+1)*(self.viewPortHeight/2)+self.yViewPort)
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
        self.framebuffer[x][y] = self.glColor(0.01176470,0.583921569,0.9882352941)  

    def glLine(self,x0, y0, x1, y1):
        x0 = round((x0+1)*(self.viewPortWidth/2)+self.xViewPort)
        y0 = round((y0+1)*(self.viewPortHeight/2)+self.yViewPort)
        x1 = round((x1+1)*(self.viewPortWidth/2)+self.xViewPort)
        y1 = round((y1+1)*(self.viewPortHeight/2)+self.yViewPort)
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        steep = dy > dx

        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)

        offset = 0 
        threshold =  dx
        y = y0

        for x in range(x0, x1):
            if steep:
                r.point(y, x)
            else:
                r.point(y, x)

            offset +=   2 *dy
            if offset >=threshold:
                y += 1 if y0 < y1 else -1
                threshold +=  2 * dx

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

    '''def load(self, filename='default.obj'):
      model = Obj(filename)

      for face in model.faces:
        vcount = len(face)

      for j in range(vcount):
        vertex_index1 = face[j][0] -1
        vertex_index1 = face[(j+1) % vcount][0] -1

        v1 = model.vertices[vertex_index1]
        v2 = model.vertices[vertex_index2]'''


r = Render()
r.glCreateWindow(100, 100)
r.glClearcolor(0.75, 0.25, 0.39)
r.glViewport(10, 10, 50, 50)
r.glVertex(-1, -1)
r.glLine(-1, -1, 1, 1)
r.glFinish()