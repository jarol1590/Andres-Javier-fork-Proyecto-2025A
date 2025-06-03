import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA
from mpi4py import MPI


from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    EFECTO,
    ACTUAL,
)

class QNodes(SIA):
   

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        # self.memoria_delta = dict()
        self.memoria_omega = dict()
        self.memoria_particiones = dict()

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.logger = SafeLogger(QNODES_STRAREGY_TAG)

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        futuro = tuple(
            (EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        mip = self.algorithm(vertices)

       
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank != 0:
            return None

        fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))
        perdida_mip, dist_marginal_mip = self.memoria_particiones[mip]

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=perdida_mip,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal_mip,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
        
    
    def algorithm(self, vertices: list[tuple[int, int]]):
        """
        Implementa el algoritmo Q para encontrar la partición óptima de un sistema que minimiza la pérdida de información, basándose en principios de submodularidad dentro de la teoría de lainformación.

        El algoritmo opera sobre un conjunto de vértices que representan nodos en diferentes tiempos del sistema (presente y futuro). La idea fundamental es construir incrementalmente grupos de nodos que, cuando se particionan, producen la menor pérdida posible de información en el sistema.

        Proceso Principal:
        -----------------
        El algoritmo comienza estableciendo dos conjuntos fundamentales: omega (W) y delta.
        Omega siempre inicia con el primer vértice del sistema, mientras que delta contiene todos los vértices restantes. Esta decisión no es arbitraria - al comenzar con un
        solo elemento en omega, podemos construir grupos de manera incremental evaluando cómo cada adición afecta la pérdida de información.

        La ejecución se desarrolla en fases, ciclos e iteraciones, donde cada fase representa un nivel diferente y conlleva a la formación de una partición candidata, cada ciclo representa un incremento de elementos al conjunto W y cada iteración determina al final cuál es el mejor elemento/cambio/delta para añadir en W.
        Fase >> Ciclo >> Iteración.

        1. Formación Incremental de Grupos:
        El algoritmo mantiene un conjunto omega que crece gradualmente en cada j-iteración. En cada paso, evalúa todos los deltas restantes para encontrar cuál, al unirse con omega produce la menor pérdida de información. Este proceso utiliza la función submodular para calcular la diferencia entre la EMD (Earth Mover's Distance) de la combinación y la EMD individual del delta evaluado.

        2. Evaluación de deltas:
        Para cada delta candidato el algoritmo:
        - Calcula su EMD individual si no está en memoria.
        - Calcula la EMD de su combinación con el conjunto omega actual
        - Determina la diferencia entre estas EMDs (el "costo" de la combinación)
        El delta que produce el menor costo se selecciona y se añade a omega.

        3. Formación de Nuevos Grupos:
        Al final de cada fase cuando omega crezca lo suficiente, el algoritmo:
        - Toma los últimos elementos de omega y delta (par candidato).
        - Los combina en un nuevo grupo
        - Actualiza la lista de vértices para la siguiente fase
        Este proceso de agrupamiento permite que el algoritmo construya particiones
        cada vez más complejas y reutilice estos "pares candidatos" para particiones en conjunto.

        Optimización y Memoria:
        ----------------------
        El algoritmo utiliza dos estructuras de memoria clave:
        - individual_memory: Almacena las EMDs y distribuciones de nodos individuales, evitando recálculos muy costosos.
        - partition_memory: Guarda las EMDs y distribuciones de las particiones completas, permitiendo comparar diferentes combinaciones de grupos teniendo en cuenta que su valor real está asociado al valor individual de su formación delta.

        La memoización es relevante puesto muchos cálculos de EMD son computacionalmente costosos y se repiten durante la ejecución del algoritmo.

        Resultado:
        ---------------
        Al terminar todas las fases, el algoritmo selecciona la partición que produjo la menor EMD global, representando la división del sistema que mejor preserva su información causal.

        Args:
            vertices (list[tuple[int, int]]): Lista de vértices donde cada uno es una
                tupla (tiempo, índice). tiempo=0 para presente (t_0), tiempo=1 para futuro (t_1).

        Returns:
            tuple[float, tuple[tuple[int, int], ...]]: El valor de pérdida en la primera posición, asociado con la partición óptima encontrada, identificada por la clave en partition_memory que produce la menor EMD.
        """
        
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        omegas_origen = [vertices[0]]
        deltas_origen = list(vertices[1:])
        vertices_fase = list(vertices)


        omegas_ciclo = omegas_origen
        deltas_ciclo = deltas_origen

        total = len(vertices_fase) - 2
        for i in range(total):
            if rank == 0:
                self.logger.debug(f"total: {total - i}")
            # Cada ciclo, repartir deltas entre procesos
            for j in range(len(deltas_ciclo) - 1):
                # Preparar lista de argumentos para cada delta
                if rank == 0:
                    args_list = [
                        (deltas_ciclo[k], omegas_ciclo)
                        for k in range(len(deltas_ciclo))
                    ]
                    # Dividir en chunks para cada proceso
                    chunk_size = (len(args_list) + size - 1) // size
                    chunks = [args_list[k*chunk_size:(k+1)*chunk_size] for k in range(size)]
                else:
                    chunks = None

                my_chunk = comm.scatter(chunks, root=0)
                my_results = [self.funcion_submodular(*args) for args in my_chunk]
                all_results = comm.gather(my_results, root=0)

                if rank == 0:
                    resultados = [item for sublist in all_results for item in sublist]
                    emd_local = float('inf')
                    indice_mip = None
                    for k, (emd_union, emd_delta, dist_marginal_delta) in enumerate(resultados):
                        emd_iteracion = emd_union - emd_delta
                        if emd_iteracion < emd_local:
                            emd_local = emd_iteracion
                            indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta

                    if indice_mip is None:
                        indice_mip = 0  # fallback

                    omegas_ciclo.append(deltas_ciclo[indice_mip])
                    deltas_ciclo.pop(indice_mip)

            # Solo el proceso 0 actualiza la memoria y los conjuntos
            if rank == 0:
                clave = (
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )
                self.memoria_particiones[tuple(clave)] = emd_particion_candidata, dist_particion_candidata

                par_candidato = (
                    [omegas_ciclo[LAST_IDX]]
                    if isinstance(omegas_ciclo[LAST_IDX], tuple)
                    else omegas_ciclo[LAST_IDX]
                ) + (
                    deltas_ciclo[LAST_IDX]
                    if isinstance(deltas_ciclo[LAST_IDX], list)
                    else deltas_ciclo
                )

                omegas_ciclo.pop()
                omegas_ciclo.append(par_candidato)
                vertices_fase = omegas_ciclo

            # Sincronizar los conjuntos para todos los procesos
            omegas_ciclo = comm.bcast(omegas_ciclo, root=0)
            deltas_ciclo = comm.bcast(deltas_ciclo, root=0)
            vertices_fase = comm.bcast(vertices_fase, root=0)

        if rank == 0:
            return min(
                self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
            )
        else:
            return None
            
        
    
    def funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
     
        emd_delta = INFTY_NEG
        temporal = [[], []]

        if isinstance(deltas, tuple):
            d_tiempo, d_indice = deltas
            temporal[d_tiempo].append(d_indice)

        else:
            for delta in deltas:
                d_tiempo, d_indice = delta
                temporal[d_tiempo].append(d_indice)

        copia_delta = self.sia_subsistema

        dims_alcance_delta = temporal[EFECTO]
        dims_mecanismo_delta = temporal[ACTUAL]

        particion_delta = copia_delta.bipartir(
            np.array(dims_alcance_delta, dtype=np.int8),
            np.array(dims_mecanismo_delta, dtype=np.int8),
        )
        vector_delta_marginal = particion_delta.distribucion_marginal()
        emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)

        # Unión #

        for omega in omegas:
            if isinstance(omega, list):
                for omg in omega:
                    o_tiempo, o_indice = omg
                    temporal[o_tiempo].append(o_indice)
            else:
                o_tiempo, o_indice = omega
                temporal[o_tiempo].append(o_indice)

        copia_union = self.sia_subsistema

        dims_alcance_union = temporal[EFECTO]
        dims_mecanismo_union = temporal[ACTUAL]

        particion_union = copia_union.bipartir(
            np.array(dims_alcance_union, dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)
 
        return emd_union, emd_delta, vector_delta_marginal

    
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))

