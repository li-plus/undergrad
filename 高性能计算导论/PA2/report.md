# 高性能计算导论 PA2

> 2017011620  计73  李家昊

### 源码

OpenMP 版本：`bfs_omp.cpp` 文件中的 `bfs_omp` 函数的源码如下， 实现思路见注释。

```cpp
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

    // examine all outgoing neighbors
    for (const int *nbr = start_nbr; nbr != end_nbr; nbr++) {
      int outgoing = *nbr;

      if (distances[outgoing] == NOT_VISITED_MARKER) {
        // neighbor is not yet visited, set its distance atomically
        if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)) {
          // add to worker's frontier queue
          int index = worker_frontier.count++;
          worker_frontier.vertices[index] = outgoing;
        }
      }
    }
  }
#pragma omp critical
{
  // merge worker frontier queues
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

    // examine all incoming neighbors
      for (const int *nbr = start_nbr; nbr != end_nbr; nbr++) {
        int incoming = *nbr;

        if (distances[incoming] == level) {
          // incoming node is in frontier set, so this node is discovered
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
  // merge worker frontier queues
  memcpy(new_frontier->vertices + new_frontier->count,
    worker_frontier.vertices, sizeof(int) * worker_frontier.count);
  new_frontier->count += worker_frontier.count;
}
  vertex_set_destroy(&worker_frontier);
}
}

void bfs_omp(Graph graph, solution *sol) {
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
```

OpenMP + MPI 版本：`bfs_omp_mpi.cpp` 文件中的 `bfs_omp_mpi` 函数的源码如下，实现思路与 OpenMP 版本类似。

```cpp
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

    // gather global new frontier
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

    // set distances of new frontier
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
```

### 优化方式

+ 实现了 Top-Down 和 Bottom-Up 混合的算法，每一轮迭代根据当前状况选择最优策略
+ 使用 `compare_and_swap` 原子操作避免数据竞争，而不是 `omp critical`
+ 并行扩展 frontier 时，每个线程独享一个 frontier 队列，而不是采用一个带锁的全局队列
+ 在合适的地方使用 `nowait`
+ 并行化几乎所有可并行的 `for` 循环

### 实验结果

串行版本：使用框架提供的 `bfs_serial` 进行评测，在 `68m.graph` 图上的运行时间为 **452.917** ms。

OpenMP 版本：使用 1, 7, 14, 28 线程在 `68m.graph` 图上 `bfs_omp` 函数的运行时间，相对 `bfs_serial` 以及相对单线程的加速比，如下表所示。

| 线程数 | 运行时间（ms） | 相对 `bfs_serial` 的加速比 | 相对单线程的加速比 |
| ------ | -------------- | -------------------------- | ------------------ |
| serial | 452.917        | 1.000                      | /                  |
| 1      | 272.189        | 1.664                      | 1.000              |
| 7      | 105.603        | 4.289                      | 2.577              |
| 14     | 60.614         | 7.472                      | 4.491              |
| 28     | **36.200**     | **12.512**                 | **7.519**          |

OpenMP + MPI 版本：使用 1x1, 1x2, 1x4, 1x14, 1x28, 2x1, 2x2, 2x4, 2x14, 2x28, 4x1, 4x2, 4x4, 4x14, 4x28 进程 （NxP 表示 N 台机器，每台机器 P 个进程）在 `68m.graph` 图上 `bfs_omp` 函数的运行时间，相对 `bfs_serial` 以及相对单进程的加速比，如下表所示。

| 进程数 | 线程数 | 运行时间（ms） | 相对 `bfs_serial` 的加速比 | 相对单进程的加速比 |
| ------ | ------ | -------------- | -------------------------- | ------------------ |
| serial | 1      | 452.917        | 1.000                      | /                  |
| 1x1    | 28     | 48.072         | 9.422                      | 1.000              |
| 1x2    | 14     | 64.904         | 6.978                      | 0.741              |
| 1x4    | 7      | 64.817         | 6.988                      | 0.742              |
| 1x14   | 2      | 88.401         | 5.123                      | 0.544              |
| 1x28   | 1      | 114.013        | 3.973                      | 0.422              |
| 2x1    | 28     | 37.669         | 12.024                     | 1.276              |
| 2x2    | 14     | 46.963         | 9.644                      | 1.024              |
| 2x4    | 7      | 50.790         | 8.917                      | 0.946              |
| 2x14   | 2      | 100.028        | 4.528                      | 0.481              |
| 2x28   | 1      | 105.383        | 4.298                      | 0.456              |
| 4x1    | 28     | **28.528**     | **15.876**                 | **1.685**          |
| 4x2    | 14     | 40.395         | 11.212                     | 1.190              |
| 4x4    | 7      | 55.845         | 8.110                      | 0.861              |
| 4x14   | 2      | 105.570        | 4.290                      | 0.455              |
| 4x28   | 1      | 108.485        | 4.175                      | 0.443              |

选取最优设置来评测 OpenMP + MPI 相对 OpenMP 的加速比，即 OpenMP 版本取 28 线程，OpenMP + MPI 版本取 4 机器 x 1 进程 x 28 线程，在 `68m.graph`，`200m.graph` 和 `500m.graph` 图上进行评测，运行时间（ms）及加速比如下表。

|              | Serial   | OpenMP  | OpenMP + MPI | MPI 相对 OpenMP 加速比 |
| ------------ | -------- | ------- | ------------ | ---------------------- |
| `68m.graph`  | 452.917  | 36.200  | 28.528       | 1.269                  |
| `200m.graph` | 4053.861 | 278.264 | 208.418      | 1.335                  |
| `500m.graph` | 9132.173 | 433.720 | 301.286      | 1.440                  |

在三个图上，OpenMP + MPI 版本相对 OpenMP 版本的加速比均超过 1.2。

