__all__ = ['VmdBezier']
import glm

class VmdBezier:
    def __init__(self):
        pass

    def SetBezier(self,x0,x1,y0,y1):
        self.m_cp1 = x0
        self.m_cp1 = glm.vec2(x0 / 127.0, y0 / 127.0)
        self.m_cp2 = glm.vec2(x1 / 127.0, y1 / 127.0)
    
    def EvalX(self,t):
        t2 = t * t
        t3 = t2 * t
        it = 1.0 - t
        it2 = it * it
        it3 = it2 * it
        x = [0, self.m_cp1.x, self.m_cp2.x, 1]
        return t3 * x[3] + 3 * t2 * it * x[2] + 3 * t * it2 * x[1] + it3 * x[0]

    def FindBezierX(self, time):
        e = 0.00001
        start = 0.0
        stop = 1.0
        t = 0.5
        x = self.EvalX(t)
        while (abs(time - x) > e):
            # print(str(time)+" "+str(x))
            if (time < x):
                stop = t
            else:  
                start = t
            t = (stop + start) * 0.5
            # print(t)
            x = self.EvalX(t)

        return t

    def EvalYfromTime(self, time):
        t = self.FindBezierX(time)
        t2 = t * t
        t3 = t2 * t
        it = 1.0 - t
        it2 = it * it
        it3 = it2 * it
        y = [0, self.m_cp1.y, self.m_cp2.y, 1]
        return t3 * y[3] + 3 * t2 * it * y[2] + 3 * t * it2 * y[1] + it3 * y[0]