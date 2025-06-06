from src.controllers.manager import Manager
import matplotlib.pyplot as plt
import numpy as np



from src.controllers.strategies.q_nodes_sec import QNodes
from src.controllers.strategies.q_nodes import QNodes as QNodesMPI

from src.controllers.strategies.q_nodes_parallel import QNodesParallel



def iniciar():

                  # 15 nodos #111111111111111
                  #20 nodos 11111111111111111111
  estado_inicial = "111111111111111" 
  condiciones =    "111111111111111"
  alcance =        "111111111111111"
  mecanismo =      "111111111111111"
  gestor_sistema = Manager(estado_inicial)

  ### Ejemplo de solución mediante módulo de fuerza bruta ###
  
  
  #descomentar segun la estrategia que se quiera probar
  
  
  #analizador_qn = QNodes(gestor_sistema)
  analizador_qn = QNodesMPI(gestor_sistema)
  #analizador_qn = QNodesParallel(gestor_sistema)
  

  # ✅ Verifica que existe TPM de 20 nodos, o créala si no
  #if not gestor_sistema.tpm_filename.exists():
    #   print(f"Archivo TPM de 20 nodos no encontrado. Generando uno nuevo...")
    #  gestor_sistema.generar_red(dimensiones=25, datos_discretos=True)

  # 🧠 Ejecutar estrategia
  #analizador_qn = QNodes(gestor_sistema)
  
  sia_uno = analizador_qn.aplicar_estrategia(condiciones, alcance, mecanismo)

  print((sia_uno)) 
    

