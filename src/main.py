from src.controllers.manager import Manager

from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # ABCDEFGHIJKLMNO #
    estado_inicial = "111111111111111"  #estado inicial del sistema
    condiciones =    "111111111111111"   #condiciones de background
    alcance =        "111111111111111"   #con cuales varaiables se quieren trabajar
    mecanismo =      "111111111111111"  #variables presentes

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_qn = QNodes(gestor_sistema)

    sia_uno = analizador_qn.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_uno)
