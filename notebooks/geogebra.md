f(x,y)=If(0≤x≤2 ∧ 0≤y≤2, 0.25, 0)
g(x,y)=If(0≤x≤2 ∧ 0≤y≤2 ∧ x+y≤2, 0.25, 0)
a = Slider(0, 2)
Segment((0,a,0),(2-a,a,0))
Polygon((0,a,0),(2-a,a,0),(2-a,a,0.25),(0,a,0.25))