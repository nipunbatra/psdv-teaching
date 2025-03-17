from manim import *

class DoubleIntegralVisualization(Scene):
    def construct(self):
        title = Text("Visualizing Double Integral for P(B)").scale(0.8).to_edge(UP)
        self.play(Write(title))

        axes = Axes(
            x_range=[0, 2.2, 1],
            y_range=[0, 2.2, 1],
            axis_config={"color": BLUE},
        ).add_coordinates()
        
        self.play(Create(axes))

        region = Polygon(
            axes.c2p(0, 0),
            axes.c2p(2, 0),
            axes.c2p(0, 2),
            color=YELLOW, fill_opacity=0.5
        )
        self.play(FadeIn(region))

        eqn_1 = MathTex("P(B) = \frac{1}{4} \int_0^2 \int_0^{2-x} dy dx").scale(0.8)
        eqn_1.next_to(title, DOWN)
        self.play(Write(eqn_1))
        self.wait(2)

        eqn_2 = MathTex("\int_0^{2-x} dy = (2-x)").scale(0.8)
        eqn_2.next_to(eqn_1, DOWN)
        self.play(Write(eqn_2))
        self.wait(2)

        eqn_3 = MathTex("\int_0^2 (2-x) dx = 2").scale(0.8)
        eqn_3.next_to(eqn_2, DOWN)
        self.play(Write(eqn_3))
        self.wait(2)

        final_eqn = MathTex("P(B) = \frac{1}{4} \times 2 = \frac{1}{2}").scale(0.8)
        final_eqn.next_to(eqn_3, DOWN)
        self.play(Write(final_eqn))
        self.wait(3)

        self.play(FadeOut(title, axes, region, eqn_1, eqn_2, eqn_3, final_eqn))
