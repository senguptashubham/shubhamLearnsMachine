import numpy as np
import matplotlib.pyplot as plt



def dtw(series_a:list, series_b:list):
  all_paths =[]
  cost_mat = []
  acc_cost_mat = []
  for b in range(len(series_b)):
    cost_row = []
    for a in range(len(series_a)):
      cost_row.append(abs(series_b[b]-series_a[a]))
    cost_mat.append(cost_row)
  for b in range(len(series_b)):
    acc_cost_row = []
    cur_path = []
    for a in range(len(series_a)):
      candidates = []
      if (b-1) >= 0 and (a-1) >= 0:
        candidates.append((acc_cost_mat[b-1][a-1], (b-1, a-1)))
      if (b-1) >= 0:
        candidates.append((acc_cost_mat[b-1][a], (b-1, a)))
      if (a-1) >= 0:
        candidates.append((acc_cost_row[a-1], (b, a-1)))
      if a == 0 and b == 0:
        cur_path.append((0,0))
        acc_cost_row.append(cost_mat[b][a] + 0)
        continue
      else:
        min_candidate = min(candidates, key=lambda item:item[0])
        cur_path.append(min_candidate[1])
      acc_cost_row.append(cost_mat[b][a] + min_candidate[0])
    acc_cost_mat.append(acc_cost_row)
    all_paths.append(cur_path)
  j = len(acc_cost_mat) - 1
  i = len(acc_cost_mat[0]) - 1
  best_path = [(j, i)]
  while j > 0 and i> 0:
    curr = all_paths[j][i]
    best_path.append(curr)
    j = curr[0]
    i = curr[1]
  while i > 0:
    i -= 1
    best_path.append((0, i))
  while j > 0:
    j -= 1
    best_path.append((j, 0))
  return cost_mat, acc_cost_mat, best_path

def visualize_dtw(series_a, series_b, cost_mat, acc_cost_mat, path):
  cost_mat = np.array(cost_mat)
  acc_cost_mat = np.array(acc_cost_mat)
  path_b, path_a = zip(*path)

  fig, (ax_series, ax_cost, ax_acc_cost) = plt.subplots(1, 3, figsize=(18, 6))

  ax_series.plot(series_a, label='series_a', marker='o')
  ax_series.plot(series_b, label='series_b', marker='o')
  ax_series.set_title('Input series')
  ax_series.legend()

  matrix_axes = ((ax_cost, cost_mat, 'Distance Matrix'), (ax_acc_cost, acc_cost_mat, 'Cumulative Distance Matrix'))
  for ax, data, title in matrix_axes:
    ax.imshow(data, origin='lower', cmap='Blues')
    for b in range(data.shape[0]):
      for a in range(data.shape[1]):
        ax.text(a, b, f'{data[b, a]:.1f}', ha='center', va='center', fontsize=8)
    ax.plot(path_a, path_b, color='red', linewidth=2, marker='o', markersize=4)
    ax.set_title(title)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))

  fig.tight_layout()
  return fig

series_a = [1, 3, 4, 9, 8, 2, 1, 5, 7, 3]
series_b = [1, 6, 2, 3, 0, 9, 4, 3, 6, 3]
cost_mat, acc_cost_mat, path = dtw(series_a, series_b)
visualize_dtw(series_a, series_b, cost_mat, acc_cost_mat, path)

plt.show()