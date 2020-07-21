'''
    Diana Ximena de León Figueroa
    Carne 18607
    SR1 Point
    Graficas por Computadora
    09 de julio de 2020
'''

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
        self.framebuffer = []

    def point(self, x, y):
        self.framebuffer[y][x] = self.color

    def glInit(self):
        pass

    def glCreateWindow(self, width, height):
        self.width = width
        self.height = height

    def glViewport(self, x, y, width, height):
        self.xViewPort = x
        self.yViewPort = y
        self.viewPortWidth = width
        self.viewPortHeight = height

    def glClear(self):
        self.framebuffer = [
            [color(0, 0, 0) for x in range(self.width)]
            for y in range(self.height)
        ]

    def glClearColor(self, r=1, g=1, b=1):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)

        self.framebuffer = [
            [color(r, g, b) for x in range(self.width)]
            for y in range(self.height)
        ]

    def glColor(self, r=0.5, g=0.5, b=0.5):
        r = round(r*255)
        g = round(g*255)
        b = round(b*255)
        self.color = color(r, g, b)

    def glVertex(self, x, y):
        X = round((x+1)*(self.viewPortWidth/2)+self.xViewPort)
        Y = round((y+1)*(self.viewPortHeight/2)+self.yViewPort)
        self.point(X, Y)

    def glLine(self,x1, y1, x2, y2):
        x1 = round((x1+1)*(self.viewPortWidth/2)+self.xViewPort)
        y1 = round((y1+1)*(self.viewPortHeight/2)+self.yViewPort)
        x2 = round((x2+1)*(self.viewPortWidth/2)+self.xViewPort)
        y2 = round((y2+1)*(self.viewPortHeight/2)+self.yViewPort)
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)

        steep = dy > dx

        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dy = abs(y2 - y1)
        dx = abs(x2 - x1)

        offset = 0 
        threshold =  dx
        y = y1

        for x in range(x1, x2):
            if steep:
                bitmap.point(y, x)
            else:
                bitmap.point(y, x)

            offset +=   2 *dy
            if offset >=threshold:
                y += 1 if y1 < y2 else -1
                threshold +=  2 * dx


    def glFinish(self, filename='out.bmp'):
        f = open(filename, 'bw')

        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        # image header
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

        # pixel data
        for x in range(self.width):
            for y in range(self.height):
                f.write(self.framebuffer[y][x])

        f.close()


bitmap = Render()

bitmap.glCreateWindow(100, 100)
bitmap.glClearColor(0.33, 0.33, 0.33)
bitmap.glViewport(10, 10, 50, 50)
bitmap.glColor(1, 0.28, 0)
bitmap.glVertex(-1, -1)
bitmap.glLine(-1,-1, 1, 1)
bitmap.glFinish()
