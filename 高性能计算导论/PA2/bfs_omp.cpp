#include "bfs_common.h"
#include "graph.h"
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_init(vertex_set *list, int count);

static inline void vertex_set_destroy(vertex_set *list) {
  free(list->vertices);
}

static void top_down_step_omp(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                              int *distances) {
  new_frontier->count = 0;

#pragma omp parallel
{
  vertex_set worker_frontier;
  vertex_set_init(&worker_frontier, g->num_nodes);

#pragma omp for nowait
  for (int i = 0; i < frontier->count; i++) {

    int node = frontier->vertices[i];

    const int *start_nbr = outgoing_begin(g, node);
    const int *end_nbr = outgoing_end(g, node);

    for (const int *nbr = start_nbr; nbr != end_nbr; nbr++) {
      int outgoing = *nbr;

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

static void bottom_up_step_omp(Graph g, vertex_set *new_frontier, int *distances, int level) {
  new_frontier->count = 0;

#pragma omp parallel
{
  vertex_set worker_frontier;
  vertex_set_init(&worker_frontier, g->num_nodes);

#pragma omp for nowait
  for (int i = 0; i < g->num_nodes; i++) {
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

void bfs_omp(Graph graph, solution *sol) {
  /** Your code ... */
  static const int ALPHA = 14;
  static const int BETA = 24;
  bool is_top_down = true;

  vertex_set _frontier;
  vertex_set *frontier = &_frontier;
  vertex_set_init(frontier, graph->num_nodes);

  vertex_set _new_frontier;
  vertex_set *new_frontier = &_new_frontier;
  vertex_set_init(new_frontier, graph->num_nodes);

  // initialize all nodes to NOT_VISITED
#pragma omp parallel for
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  int m_u = graph->num_edges;   // edges connected to unvisited nodes

  int level = 0;
  while (frontier->count != 0) {
    // compute edges connected to frontier
    int m_f = 0;
#pragma omp parallel for reduction(+:m_f)
    for (int i = 0; i < frontier->count; i++) {
      int node = frontier->vertices[i];
      m_f += outgoing_size(graph, node);
    }
    // decrease edges connected to unvisited nodes
    m_u -= m_f;

    // check whether need to switch policy
    // slides: https://people.csail.mit.edu/jshun/6886-s18/lectures/lecture4-1.pdf
    // paper: https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf
    if (is_top_down) {
      // if m_f > (m_u / alpha), then switch to bottom_up
      // m_f: edges adjacent to frontier
      // m_u: edges adjacent to unvisited nodes
      if (ALPHA * m_f > m_u) {
        is_top_down = false;
      }
    } else {
      // if n_f < (n / beta), then switch to top_down
      // n_f: vertices in frontier
      // n: total vertices
      if (BETA * frontier->count < graph->num_nodes) {
        is_top_down = true;
      }
    }

    if (is_top_down) {
      top_down_step_omp(graph, frontier, new_frontier, sol->distances);
    } else {
      bottom_up_step_omp(graph, new_frontier, sol->distances, level);
    }

    std::swap(frontier, new_frontier);

    level++;
  }
  vertex_set_destroy(frontier);
  vertex_set_destroy(new_frontier);
}
