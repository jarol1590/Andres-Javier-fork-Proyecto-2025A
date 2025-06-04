from src.controllers.manager import Manager

from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.q_nodes_parallel import QNodesParallel


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #111111111111111
    estado_inicial = "000000000000000" #15 bits
    condiciones =    "111111111111111"
    alcance =        "111111111111111"
    mecanismo =      "111111111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    #analizador_qn = QNodes(gestor_sistema)
    analizador_qn = QNodesParallel(gestor_sistema)


    sia_uno = analizador_qn.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_uno)
