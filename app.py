import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp

# Configuração inicial da página
st.set_page_config(page_title="Geometria Analítica: Retas e Planos", layout="centered")


# Funções auxiliares
def plot_reta_2d(a, b):
    x_vals = np.linspace(-10, 10, 400)
    y_vals = a * x_vals + b

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label=f'y = {a}x + {b}', color='blue')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, which='both')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    return fig


def plot_reta_parametrica_2d(x0, y0, vx, vy, t, show_vector=False):
    x = x0 + t * vx
    y = y0 + t * vy

    fig, ax = plt.subplots()
    ax.plot(x, y, label=f'Reta Paramétrica', color='red')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, which='both')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if show_vector:
        ax.quiver(x0, y0, vx, vy, angles='xy', scale_units='xy', scale=1, color='green', label='Vetor Diretor')

    ax.legend()
    return fig


def plot_reta_3d(a, b, c, d):
    t = np.linspace(-10, 10, 100)
    x = t
    y = a * t + b
    z = c * t + d

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='lines',
                                       line=dict(color='red', width=2))])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      width=700, height=700)
    return fig


def plot_plano_3d(A, B, C, D):
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)

    if C != 0:
        Z = (-D - A * X - B * Y) / C
    else:
        Z = np.full_like(X, np.nan)  # Plano vertical

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                      width=700, height=700)
    return fig


# Páginas do aplicativo
def pagina_inicial():
    st.title('Geometria Analítica - Retas e Planos')
    st.write("""
    Bem-vindo ao aplicativo interativo sobre Geometria Analítica focado em Retas e Planos!
    Use o menu na barra lateral para navegar entre as diferentes seções.

    Este aplicativo inclui:
    - Visualizações interativas de retas em 2D e 3D
    - Estudo de planos no espaço 3D
    - Exercícios interativos com soluções detalhadas
    - Dúvidas comuns e suas respostas
    - Ferramentas interativas para exploração de conceitos

    Explore as diferentes seções e divirta-se aprendendo!
    """)


def retas_2d():
    st.title('Retas no Plano 2D')

    st.header("Equação Reduzida da Reta: y = ax + b")
    a = st.slider('Coeficiente angular (a)', -10.0, 10.0, 1.0, key='reta_2d_a')
    b = st.slider('Intercepto (b)', -10.0, 10.0, 0.0, key='reta_2d_b')

    st.latex(r"y = ax + b")
    fig = plot_reta_2d(a, b)
    st.pyplot(fig)

    st.write("""
    No gráfico acima, a reta `y = ax + b` está sendo representada. Altere o valor de `a` para observar como a inclinação da reta muda.
    """)

    st.header("Equação Paramétrica da Reta")
    st.write("""
    A equação paramétrica de uma reta é uma maneira de representar as coordenadas de pontos em uma reta em função de um parâmetro `t`.
    Ela tem a seguinte forma:
    """)
    st.latex(r"x = x_0 + t \cdot v_x, \quad y = y_0 + t \cdot v_y")

    x0 = st.slider('Ponto de referência x_0', -10.0, 10.0, 0.0, key='param_2d_x0')
    y0 = st.slider('Ponto de referência y_0', -10.0, 10.0, 0.0, key='param_2d_y0')
    vx = st.slider('Componente x do vetor diretor', -10.0, 10.0, 1.0, key='param_2d_vx')
    vy = st.slider('Componente y do vetor diretor', -10.0, 10.0, 1.0, key='param_2d_vy')

    show_vector = st.checkbox('Mostrar vetor diretor', key='param_2d_show_vector')

    t_vals = np.linspace(-10, 10, 400)
    fig = plot_reta_parametrica_2d(x0, y0, vx, vy, t_vals, show_vector)
    st.pyplot(fig)


def retas_3d():
    st.title('Retas e Planos no Espaço 3D')

    st.header("Equação Paramétrica da Reta no Espaço")
    st.write("""
    A equação paramétrica de uma reta no espaço 3D é dada por:
    """)
    st.latex(r"x = x_0 + at, \quad y = y_0 + bt, \quad z = z_0 + ct")

    x0 = st.slider('Ponto de referência x_0', -10.0, 10.0, 0.0, key='param_3d_x0')
    y0 = st.slider('Ponto de referência y_0', -10.0, 10.0, 0.0, key='param_3d_y0')
    z0 = st.slider('Ponto de referência z_0', -10.0, 10.0, 0.0, key='param_3d_z0')
    a = st.slider('Componente a do vetor diretor', -10.0, 10.0, 1.0, key='param_3d_a')
    b = st.slider('Componente b do vetor diretor', -10.0, 10.0, 1.0, key='param_3d_b')
    c = st.slider('Componente c do vetor diretor', -10.0, 10.0, 1.0, key='param_3d_c')

    fig = plot_reta_3d(a, b, c, z0)
    st.plotly_chart(fig)

    st.header("Equação Geral do Plano")
    st.write("""
    A equação geral de um plano no espaço 3D é dada por:
    """)
    st.latex(r"Ax + By + Cz + D = 0")

    A = st.slider('Coeficiente A', -10.0, 10.0, 1.0, key='plano_A')
    B = st.slider('Coeficiente B', -10.0, 10.0, 1.0, key='plano_B')
    C = st.slider('Coeficiente C', -10.0, 10.0, 1.0, key='plano_C')
    D = st.slider('Coeficiente D', -10.0, 10.0, 0.0, key='plano_D')

    fig = plot_plano_3d(A, B, C, D)
    st.plotly_chart(fig)


def exercicios():
    st.title('Exercícios Interativos')

    materia = st.selectbox("Escolha a matéria", ["Retas 2D", "Retas 3D", "Planos"])

    if materia == "Retas 2D":
        exercicio = st.selectbox("Escolha o exercício", [
            "Equação da Reta",
            "Interseção de Retas",
            "Distância Ponto-Reta",
            "Ângulo entre Retas"
        ])

        if exercicio == "Equação da Reta":
            st.subheader('Exercício: Equação da Reta')
            st.write(
                "Dada a reta que passa pelos pontos (1, 2) e (4, 8), determine a equação da reta na forma y = ax + b.")

            resposta_a = st.text_input('Digite o valor de a:')
            resposta_b = st.text_input('Digite o valor de b:')

            if st.button('Verificar Resposta'):
                a_correto = 2
                b_correto = 0
                if resposta_a and resposta_b:
                    try:
                        a_usuario = float(resposta_a)
                        b_usuario = float(resposta_b)
                        if abs(a_usuario - a_correto) < 0.01 and abs(b_usuario - b_correto) < 0.01:
                            st.success("Correto! A equação da reta é y = 2x + 0 ou simplesmente y = 2x.")
                        else:
                            st.error("Incorreto. Tente novamente.")
                    except ValueError:
                        st.error("Por favor, insira valores numéricos válidos.")
                else:
                    st.warning("Por favor, preencha ambos os campos.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Coeficiente angular: } a &= \frac{y_2 - y_1}{x_2 - x_1} = \frac{8 - 2}{4 - 1} = \frac{6}{3} = 2 \\
                2) \text{ Equação da reta: } y &= ax + b \\
                3) \text{ Substituindo um ponto: } 2 &= 2(1) + b \\
                4) \text{ Resolvendo para b: } b &= 2 - 2 = 0 \\
                5) \text{ Equação final: } y &= 2x + 0 \text{ ou } y = 2x
                \end{aligned}
                ''')

        elif exercicio == "Interseção de Retas":
            st.subheader('Exercício: Interseção de Retas')
            st.write("Encontre o ponto de interseção das retas y = 2x + 1 e y = -x + 7.")

            resposta_x = st.text_input('Digite a coordenada x do ponto de interseção:')
            resposta_y = st.text_input('Digite a coordenada y do ponto de interseção:')

            if st.button('Verificar Resposta'):
                if resposta_x and resposta_y:
                    try:
                        x = float(resposta_x)
                        y = float(resposta_y)
                        if abs(x - 2) < 0.01 and abs(y - 5) < 0.01:
                            st.success("Correto! O ponto de interseção é (2, 5).")
                        else:
                            st.error("Incorreto. Tente novamente.")
                    except ValueError:
                        st.error("Por favor, insira valores numéricos válidos.")
                else:
                    st.warning("Por favor, preencha ambos os campos.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Igualamos as equações: } 2x + 1 &= -x + 7 \\
                2) \text{ Resolvemos para x: } 3x &= 6 \\
                &x = 2 \\
                3) \text{ Substituímos em qualquer equação: } y &= 2(2) + 1 = 5 \\
                4) \text{ O ponto de interseção é } &(2, 5)
                \end{aligned}
                ''')

        elif exercicio == "Distância Ponto-Reta":
            st.subheader('Exercício: Distância Ponto-Reta')
            st.write("Calcule a distância do ponto P(2, 3) à reta 3x - 4y + 5 = 0.")

            resposta = st.text_input('Digite a distância:')

            if st.button('Verificar Resposta'):
                if resposta:
                    try:
                        d = float(resposta)
                        if abs(d - 1.4) < 0.1:
                            st.success("Correto! A distância é aproximadamente 1,4 unidades.")
                        else:
                            st.error("Incorreto. Tente novamente.")
                    except ValueError:
                        st.error("Por favor, insira um valor numérico válido.")
                else:
                    st.warning("Por favor, preencha o campo da resposta.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Fórmula da distância: } d &= \frac{|Ax_0 + By_0 + C|}{\sqrt{A^2 + B^2}} \\
                2) \text{ Substituindo os valores: } d &= \frac{|3(2) - 4(3) + 5|}{\sqrt{3^2 + (-4)^2}} \\
                3) \text{ Calculando: } d &= \frac{|6 - 12 + 5|}{\sqrt{9 + 16}} = \frac{1}{\sqrt{25}} = \frac{1}{5} \\
                4) \text{ Resultado final: } d &\approx 1,4 \text{ unidades}
                \end{aligned}
                ''')

        elif exercicio == "Ângulo entre Retas":
            st.subheader('Exercício: Ângulo entre Retas')
            st.write("Calcule o ângulo entre as retas y = 2x + 1 e y = -x/2 + 3.")

            resposta = st.text_input('Digite o ângulo em graus:')

            if st.button('Verificar Resposta'):
                if resposta:
                    try:
                        angulo = float(resposta)
                        if abs(angulo - 116.57) < 1:
                            st.success("Correto! O ângulo é aproximadamente 116,57°.")
                        else:
                            st.error("Incorreto. Tente novamente.")
                    except ValueError:
                        st.error("Por favor, insira um valor numérico válido.")
                else:
                    st.warning("Por favor, preencha o campo da resposta.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Fórmula do ângulo: } \tan \theta &= \left|\frac{m_1 - m_2}{1 + m_1m_2}\right| \\
                2) \text{ Coeficientes angulares: } m_1 &= 2, m_2 = -\frac{1}{2} \\
                3) \text{ Substituindo: } \tan \theta &= \left|\frac{2 - (-\frac{1}{2})}{1 + 2(-\frac{1}{2})}\right| = \left|\frac{2.5}{0}\right| \\
                4) \text{ Calculando: } \theta &= \arctan(2.5) \\
                5) \text{ Resultado final: } \theta &\approx 116.57°
                \end{aligned}
                ''')

    elif materia == "Retas 3D":
        exercicio = st.selectbox("Escolha o exercício", [
            "Equação Paramétrica da Reta",
            "Interseção de Retas no Espaço",
            "Distância entre Retas Reversas"
        ])

        if exercicio == "Equação Paramétrica da Reta":
            st.subheader('Exercício: Equação Paramétrica da Reta')
            st.write("Determine a equação paramétrica da reta que passa pelos pontos A(1, 2, 3) e B(4, 6, 5).")

            resposta_x = st.text_input('Digite a equação para x(t):')
            resposta_y = st.text_input('Digite a equação para y(t):')
            resposta_z = st.text_input('Digite a equação para z(t):')

            if st.button('Verificar Resposta'):
                if resposta_x and resposta_y and resposta_z:
                    x_correto = "1 + 3t"
                    y_correto = "2 + 4t"
                    z_correto = "3 + 2t"
                    if resposta_x.replace(" ", "") == x_correto.replace(" ", "") and \
                            resposta_y.replace(" ", "") == y_correto.replace(" ", "") and \
                            resposta_z.replace(" ", "") == z_correto.replace(" ", ""):
                        st.success("Correto! A equação paramétrica da reta é x = 1 + 3t, y = 2 + 4t, z = 3 + 2t.")
                    else:
                        st.error("Incorreto. Tente novamente.")
                else:
                    st.warning("Por favor, preencha todos os campos.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Vetor diretor } \vec{v} &= B - A = (4-1, 6-2, 5-3) = (3, 4, 2) \\
                2) \text{ Ponto de referência: } A &= (1, 2, 3) \\
                3) \text{ Equação paramétrica: } \\
                x &= x_A + v_x \cdot t = 1 + 3t \\
                y &= y_A + v_y \cdot t = 2 + 4t \\
                z &= z_A + v_z \cdot t = 3 + 2t
                \end{aligned}
                ''')

        # Adicione mais exercícios para Retas 3D aqui

    elif materia == "Planos":
        exercicio = st.selectbox("Escolha o exercício", [
            "Equação Geral do Plano",
            "Interseção de Três Planos",
            "Distância Ponto-Plano"
        ])

        if exercicio == "Equação Geral do Plano":
            st.subheader('Exercício: Equação Geral do Plano')
            st.write("Determine a equação geral do plano que passa pelos pontos A(1, 2, 3), B(2, 3, 1) e C(0, 1, 4).")

            resposta = st.text_input('Digite a equação do plano na forma Ax + By + Cz + D = 0:')

            if st.button('Verificar Resposta'):
                if resposta:
                    # A resposta correta é 3x - 5y + z - 4 = 0
                    resposta_correta = "3x-5y+z-4=0"
                    if resposta.replace(" ", "").lower() == resposta_correta:
                        st.success("Correto! A equação do plano é 3x - 5y + z - 4 = 0.")
                    else:
                        st.error("Incorreto. Tente novamente.")
                else:
                    st.warning("Por favor, preencha o campo da resposta.")

            if st.button('Ver Solução'):
                st.write("Solução:")
                st.latex(r'''
                \begin{aligned}
                1) \text{ Vetores } \vec{AB} &= (1, 1, -2) \text{ e } \vec{AC} = (-1, -1, 1) \\
                2) \text{ Normal } \vec{n} &= \vec{AB} \times \vec{AC} = (3, -5, 1) \\
                3) \text{ Equação do plano: } &Ax + By + Cz + D = 0 \\
                &3x - 5y + z + D = 0 \\
                4) \text{ Usando A(1, 2, 3): } &3(1) - 5(2) + 3 + D = 0 \\
                &3 - 10 + 3 + D = 0 \\
                &D = 4 \\
                5) \text{ Equação final: } &3x - 5y + z - 4 = 0
                \end{aligned}
                ''')

        # Adicione mais exercícios para Planos aqui


def duvidas_comuns():
    st.title('Dúvidas Comuns')

    duvida = st.selectbox("Escolha uma dúvida", [
        "O que é o coeficiente angular?",
        "Como encontrar a equação de uma reta perpendicular?",
        "O que é um vetor normal de um plano?",
        "Como calcular a distância entre um ponto e um plano?"
    ])

    if duvida == "O que é o coeficiente angular?":
        st.write("""
        O coeficiente angular de uma reta é a tangente do ângulo que a reta forma com o eixo x positivo.
        Ele representa a inclinação da reta e pode ser calculado como:
        """)
        st.latex(r"m = \frac{y_2 - y_1}{x_2 - x_1}")
        st.write("onde (x₁, y₁) e (x₂, y₂) são dois pontos distintos na reta.")

    elif duvida == "Como encontrar a equação de uma reta perpendicular?":
        st.write("""
        Para encontrar a equação de uma reta perpendicular a uma reta dada:
        1. O produto dos coeficientes angulares de retas perpendiculares é -1.
        2. Se a reta original tem coeficiente angular m, a reta perpendicular terá coeficiente angular -1/m.
        3. Use um ponto conhecido e o novo coeficiente angular para determinar a equação.
        """)
        st.latex(r"m_1 \cdot m_2 = -1")
        st.write("onde m₁ e m₂ são os coeficientes angulares das retas perpendiculares.")

    elif duvida == "O que é um vetor normal de um plano?":
        st.write("""
        Um vetor normal de um plano é um vetor perpendicular ao plano. Na equação geral do plano Ax + By + Cz + D = 0,
        o vetor (A, B, C) é um vetor normal ao plano. Propriedades importantes:
        1. O vetor normal é perpendicular a qualquer vetor contido no plano.
        2. O vetor normal pode ser usado para calcular ângulos entre planos.
        3. O vetor normal é crucial para determinar a orientação do plano no espaço.
        """)
        st.latex(r"\vec{n} = (A, B, C)")

    elif duvida == "Como calcular a distância entre um ponto e um plano?":
        st.write("""
        Para calcular a distância d entre um ponto P(x₀, y₀, z₀) e um plano Ax + By + Cz + D = 0:
        """)
        st.latex(r"d = \frac{|Ax_0 + By_0 + Cz_0 + D|}{\sqrt{A^2 + B^2 + C^2}}")
        st.write("""
        Esta fórmula dá a menor distância do ponto ao plano. O numerador representa a substituição
        das coordenadas do ponto na equação do plano, e o denominador normaliza o resultado.
        """)


def calculadora_vetorial():
    st.title('Calculadora Vetorial')
    st.write("Realize operações com vetores em 2D e 3D.")

    dimensao = st.radio("Escolha a dimensão", ["2D", "3D"])

    if dimensao == "2D":
        v1 = [st.number_input("v1_x", value=1.0), st.number_input("v1_y", value=0.0)]
        v2 = [st.number_input("v2_x", value=0.0), st.number_input("v2_y", value=1.0)]

        if st.button("Calcular"):
            soma = [v1[0] + v2[0], v1[1] + v2[1]]
            produto_escalar = v1[0] * v2[0] + v1[1] * v2[1]

            st.write(f"Soma: {soma}")
            st.write(f"Produto Escalar: {produto_escalar}")

    elif dimensao == "3D":
        v1 = [st.number_input("v1_x", value=1.0), st.number_input("v1_y", value=0.0),
              st.number_input("v1_z", value=0.0)]
        v2 = [st.number_input("v2_x", value=0.0), st.number_input("v2_y", value=1.0),
              st.number_input("v2_z", value=0.0)]

        if st.button("Calcular"):
            soma = [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]
            produto_escalar = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
            produto_vetorial = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0]
            ]

            st.write(f"Soma: {soma}")
            st.write(f"Produto Escalar: {produto_escalar}")
            st.write(f"Produto Vetorial: {produto_vetorial}")


def transformacoes_geometricas():
    st.title('Transformações Geométricas')
    st.write("Explore transformações geométricas em 2D.")

    # Criar um ponto inicial
    x = st.number_input("Coordenada x do ponto", value=1.0)
    y = st.number_input("Coordenada y do ponto", value=1.0)

    # Escolher a transformação
    transformacao = st.selectbox("Escolha a transformação", [
        "Translação",
        "Rotação",
        "Escala"
    ])

    if transformacao == "Translação":
        dx = st.number_input("Deslocamento em x", value=2.0)
        dy = st.number_input("Deslocamento em y", value=2.0)

        x_novo = x + dx
        y_novo = y + dy

    elif transformacao == "Rotação":
        angulo = st.number_input("Ângulo de rotação (graus)", value=45.0)
        angulo_rad = np.radians(angulo)

        x_novo = x * np.cos(angulo_rad) - y * np.sin(angulo_rad)
        y_novo = x * np.sin(angulo_rad) + y * np.cos(angulo_rad)

    elif transformacao == "Escala":
        sx = st.number_input("Fator de escala em x", value=2.0)
        sy = st.number_input("Fator de escala em y", value=2.0)

        x_novo = x * sx
        y_novo = y * sy

    # Plotar o resultado
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro', label='Ponto Original')
    ax.plot(x_novo, y_novo, 'bo', label='Ponto Transformado')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    st.pyplot(fig)


def simulador_colisoes():
    st.title('Simulador de Colisões')
    st.write("Simule colisões entre objetos em 2D usando conceitos de geometria analítica.")

    # Configurações da simulação
    v1 = st.slider("Velocidade do objeto 1", 0.0, 10.0, 5.0)
    v2 = st.slider("Velocidade do objeto 2", 0.0, 10.0, 3.0)
    angulo = st.slider("Ângulo de colisão (graus)", 0, 180, 45)

    # Cálculos da colisão (simplificados)
    angulo_rad = np.radians(angulo)
    v1x = v1 * np.cos(angulo_rad)
    v1y = v1 * np.sin(angulo_rad)
    v2x = v2
    v2y = 0

    v1x_final = ((v1x - v2x) + v2x) / 2
    v2x_final = ((v2x - v1x) + v1x) / 2
    v1y_final = v1y
    v2y_final = 0

    # Visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Antes da colisão
    ax1.quiver(0, 0, v1x, v1y, angles='xy', scale_units='xy', scale=1, color='r', label='Objeto 1')
    ax1.quiver(0, 0, v2x, v2y, angles='xy', scale_units='xy', scale=1, color='b', label='Objeto 2')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.legend()
    ax1.set_title("Antes da Colisão")

    # Depois da colisão
    ax2.quiver(0, 0, v1x_final, v1y_final, angles='xy', scale_units='xy', scale=1, color='r', label='Objeto 1')
    ax2.quiver(0, 0, v2x_final, v2y_final, angles='xy', scale_units='xy', scale=1, color='b', label='Objeto 2')
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.legend()
    ax2.set_title("Depois da Colisão")

    st.pyplot(fig)


# Menu principal
st.sidebar.title("Menu de Navegação")
pagina = st.sidebar.radio("Selecione a página", [
    "Início",
    "Retas 2D",
    "Retas e Planos 3D",
    "Exercícios",
    "Dúvidas Comuns",
    "Calculadora Vetorial",
    "Transformações Geométricas",
    "Simulador de Colisões"
])

# Navegação entre páginas
if pagina == "Início":
    pagina_inicial()
elif pagina == "Retas 2D":
    retas_2d()
elif pagina == "Retas e Planos 3D":
    retas_3d()
elif pagina == "Exercícios":
    exercicios()
elif pagina == "Dúvidas Comuns":
    duvidas_comuns()
elif pagina == "Calculadora Vetorial":
    calculadora_vetorial()
elif pagina == "Transformações Geométricas":
    transformacoes_geometricas()
elif pagina == "Simulador de Colisões":
    simulador_colisoes()

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info(
    "Este aplicativo foi desenvolvido como uma ferramenta educacional "
    "para auxiliar no estudo de Geometria Analítica. "
    "Para mais informações, entre em contato com o desenvolvedor."
)

# Opcional: Adicionar um botão para reportar bugs ou dar feedback
if st.sidebar.button("Reportar um Bug / Dar Feedback"):
    st.sidebar.write("Por favor, envie um e-mail para: johnlopes360@gmail.com")

# Informações sobre a versão do aplicativo
st.sidebar.text("Versão do App: 1.0.0")
