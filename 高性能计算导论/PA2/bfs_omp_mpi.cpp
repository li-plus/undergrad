#include "bfs_common.h"
#include "graph.h"
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <algorithm>

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_init(vertex_set *list, int count);

static inline void vertex_set_destroy(vertex_set *list) {
  free(list->vertices);
}

static void top_down_step_omp(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                              int *distances, int start_node, int end_node) {
  new_frontier->count = 0;

#pragma omp parallel
{
  vertex_set worker_frontier;
  vertex_set_init(&worker_frontier, end_node - start_node);

#pragma omp for nowait
  for (int i = 0; i < frontier->count; i++) {
    int node = frontier->vertices[i];

    const int *start_nbr = outgoing_begin(g, node);
    const int *end_nbr = outgoing_end(g, node);

    for (const int *nbr = start_nbr; nbr != end_nbr; nbr++) {
      int outgoing = *nbr;

      if (!(start_node <= outgoing && outgoing < end_node)) {
        continue;
      }

      if (distances[outgoing] == NOT_VISITED_MARKER) {
        if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
          int index = worker_frontier.count++;
          worker_frontier.vertices[index] = outgoing;
        }
      }
    }
  }
#pragma omp critical
{
  memcpy(new_frontier->vertices + new_frontier->count,
    worker_frontier.vertices, sizeof(int) * worker_frontier.count);
  new_frontier->count += worker_frontier.count;
}
  vertex_set_destroy(&worker_frontier);
}
}

static void bottom_up_step_omp(Graph g, vertex_set *new_frontier, int *distances, int level,
                               int start_node, int end_node) {
  new_frontier->count = 0;

#pragma omp parallel
{
  vertex_set worker_frontier;
  vertex_set_init(&worker_frontier, end_node - start_node);

#pragma omp for nowait
  for (int i = start_node; i < end_node; i++) {
    if (distances[i] == NOT_VISITED_MARKER) {
      const int *start_nbr = incoming_begin(g, i);
      const int *end_nbr = incoming_end(g, i);

      for (const int *nbr = start_nbr; nbr != end_nbr; nbr++) {
        int incoming = *nbr;

        if (distances[incoming] == level) {
          distances[i] = level + 1;
          int index = worker_frontier.count++;
          worker_frontier.vertices[index] = i;
          break;
        }
      }
    }
  }
#pragma omp critical
{
  memcpy(new_frontier->vertices + new_frontier->count,
    worker_frontier.vertices, sizeof(int) * worker_frontier.count);
  new_frontier->count += worker_frontier.count;
}
  vertex_set_destroy(&worker_frontier);
}
}

void bfs_omp_mpi(Graph graph, solution *sol) {
  /** Your code ... */
  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  static const int ALPHA = 14;
  static const int BETA = 24;
  bool is_top_down = true;

  int mean_size = (graph->num_nodes + nprocs - 1) / nprocs;
  int start_node = rank * mean_size;
  int end_node = (rank == nprocs - 1) ? graph->num_nodes : start_node + mean_size;

  int *new_frontier_starts = new int[nprocs];
  int *new_frontier_counts = new int[nprocs];

  // global frontier
  vertex_set _frontier;
  vertex_set *frontier = &_frontier;
  vertex_set_init(frontier, graph->num_nodes);

  // local new frontier
  vertex_set _new_frontier;
  vertex_set *new_frontier = &_new_frontier;
  vertex_set_init(new_frontier, graph->num_nodes);

#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }
  sol->distances[ROOT_NODE_ID] = 0;

  frontier->vertices[frontier->count++] = ROOT_NODE_ID;

  int m_f = outgoing_size(graph, ROOT_NODE_ID);
  int m_u = graph->num_edges - m_f;

  int level = 0;

  while (true) {
    if (is_top_down) {
      if (ALPHA * m_f > m_u) {
        is_top_down = false;
      }
    } else {
      if (BETA * frontier->count < graph->num_nodes) {
        is_top_down = true;
      }
    }

    if (is_top_down) {
      top_down_step_omp(graph, frontier, new_frontier, sol->distances, start_node, end_node);
    } else {
      bottom_up_step_omp(graph, new_frontier, sol->distances, level, start_node, end_node);
    }

    m_f = 0;
#pragma omp parallel for reduction(+:m_f)
    for (int i = 0; i < new_frontier->count; i++) {
      int node = new_frontier->vertices[i];
      m_f += outgoing_size(graph, node);
    }
    MPI_Allreduce(MPI_IN_PLACE, &m_f, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    m_u -= m_f;

    MPI_Allgather(&new_frontier->count, 1, MPI_INT,
      new_frontier_counts, 1, MPI_INT, MPI_COMM_WORLD);

    new_frontier_starts[rank] = 0;
    int curr_start = new_frontier_counts[rank];
    for (int i = 0; i < nprocs; i++) {
      if (i != rank) {
        new_frontier_starts[i] = curr_start;
        curr_start += new_frontier_counts[i];
      }
    }

    MPI_Allgatherv(new_frontier->vertices, new_frontier->count, MPI_INT,
      frontier->vertices, new_frontier_counts, new_frontier_starts, MPI_INT, MPI_COMM_WORLD);

    frontier->count = 0;
    for (int i = 0; i < nprocs; i++) {
      frontier->count += new_frontier_counts[i];
    }

    if (frontier->count == 0) {
      break;
    }

    level++;

#pragma omp parallel for
    for (int i = new_frontier->count; i < frontier->count; i++) {
      int node = frontier->vertices[i];
      sol->distances[node] = level;
    }
  }

  delete[] new_frontier_counts;
  delete[] new_frontier_starts;

  vertex_set_destroy(new_frontier);
  vertex_set_destroy(frontier);
}
