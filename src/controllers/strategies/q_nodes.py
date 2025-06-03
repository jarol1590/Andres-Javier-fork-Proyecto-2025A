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
    """
    Clase QNodes para el análisis de redes mediante el algoritmo Q.

    Esta clase implementa un gestor principal para el análisis de redes que utiliza
    el algoritmo Q para encontrar la partición óptima que minimiza la
    pérdida de información en el sistema. Hereda de la clase base SIA (Sistema de
    Información Activo) y proporciona funcionalidades para analizar la estructura
    y dinámica de la red.

    Args:
    ----
        config (Loader):
            Instancia de la clase Loader que contiene la configuración del sistema
            y los parámetros necesarios para el análisis.

    Attributes:
    ----------
        m (int):
            Número de elementos en el conjunto de purview (vista).

        n (int):
            Número de elementos en el conjunto de mecanismos.

        tiempos (tuple[np.ndarray, np.ndarray]):
            Tupla de dos arrays que representan los tiempos para los estados
            actual y efecto del sistema.

        etiquetas (list[tuple]):
            Lista de tuplas conteniendo las etiquetas para los nodos,
            con versiones en minúsculas y mayúsculas del abecedario.

        vertices (set[tuple]):
            Conjunto de vértices que representan los nodos de la red,
            donde cada vértice es una tupla (tiempo, índice).

        memoria (dict):
            Diccionario para almacenar resultados intermedios y finales
            del análisis (memoización).

        logger:
            Instancia del logger configurada para el análisis Q.

    Methods:
    -------
        run(condicion, purview, mechanism):
            Ejecuta el análisis principal de la red con las condiciones,
            purview y mecanismo especificados.

        algorithm(vertices):
            Implementa el algoritmo Q para encontrar la partición
            óptima del sistema.

        funcion_submodular(deltas, omegas):
            Calcula la función submodular para evaluar particiones candidatas.

        view_solution(mip):
            Visualiza la solución encontrada en términos de las particiones
            y sus valores asociados.

        nodes_complement(nodes):
            Obtiene el complemento de un conjunto de nodos respecto a todos
            los vértices del sistema.

    Notes:
    -----
    - La clase implementa una versión secuencial del algoritmo Q para encontrar la partición que minimiza la pérdida de información.
    - Utiliza memoización para evitar recálculos innecesarios durante el proceso.
    - El análisis se realiza considerando dos tiempos: actual (presente) y
      efecto (futuro).
    """

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

        #

        futuro = tuple(
            (EFECTO, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )
        # ( (1,0)=A (1,1)=B (1,2)=C #

        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )  #
        # ( (0,0)=a (0,1)=b (0,2)=c #

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
        """
        Evalúa el impacto de combinar el conjunto de nodos individual delta y su agrupación con el conjunto omega, calculando la diferencia entre EMD (Earth Mover's Distance) de las configuraciones, en conclusión los nodos delta evaluados individualmente y su combinación con el conjunto omega.

        El proceso se realiza en dos fases principales:

        1. Evaluación Individual:
           - Crea una copia del estado temporal del subsistema.
           - Activa los nodos delta en su tiempo correspondiente (presente/futuro).
           - Si el delta ya fue evaluado antes, recupera su EMD y distribución marginal de memoria
           - Si no, ha de:
             * Identificar dimensiones activas en presente y futuro.
             * Realiza bipartición del subsistema con esas dimensiones.
             * Calcular la distribución marginal y EMD respecto al subsistema.
             * Guarda resultados en memoria para seguro un uso futuro.

        2. Evaluación Combinada:
           - Sobre la misma copia temporal, activa también los nodos omega.
           - Calcula dimensiones activas totales (delta + omega).
           - Realiza bipartición del subsistema completo.
           - Obtiene EMD de la combinación.

        Args:
            deltas: Un nodo individual (tupla) o grupo de nodos (lista de tuplas)
                   donde cada tupla está identificada por su (tiempo, índice), sea el tiempo t_0 identificado como 0, t_1 como 1 y, el índice hace referencia a las variables/dimensiones habilitadas para operaciones de substracción/marginalización sobre el subsistema, tal que genere la partición.
            omegas: Lista de nodos ya agrupados, puede contener tuplas individuales
                   o listas de tuplas para grupos formados por los pares candidatos o más uniones entre sí (grupos candidatos).

        Returns:
            tuple: (
                EMD de la combinación omega y delta,
                EMD del delta individual,
                Distribución marginal del delta individual
            )
            Esto lo hice así para hacer almacenamiento externo de la emd individual y su distribución marginal en las particiones candidatas.
        """
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