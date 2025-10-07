"""Utility functions for RecSim environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def aggregate_video_cluster_metrics(responses, metrics, info=None):
    """Aggregates the video cluster metrics with one step responses.

    Args:
      responses: a dictionary of names, observed responses.
      metrics: A dictionary mapping from metric_name to its value in float.
      info: Additional info for computing metrics (ignored here)

    Returns:
      A dictionary storing metrics after aggregation.
    """
    del info  # Unused.
    is_clicked = False
    metrics['impression'] += 1

    for response in responses:
        if not response['click']:
            continue
        is_clicked = True
        metrics['click'] += 1
        metrics['quality'] += response['quality']
        cluster_id = response['cluster_id']
        metrics['cluster_watch_count_cluster_%d' % cluster_id] += 1

    if not is_clicked:
        metrics['cluster_watch_count_no_click'] += 1
    return metrics


def write_video_cluster_metrics(metrics, add_summary_fn):
    """Writes average video cluster metrics using add_summary_fn."""
    add_summary_fn('CTR', metrics['click'] / metrics['impression'])
    if metrics['click'] > 0:
        add_summary_fn('AverageQuality', metrics['quality'] / metrics['click'])
    for k in metrics:
        prefix = 'cluster_watch_count_cluster_'
        if k.startswith(prefix):
            add_summary_fn('cluster_watch_count_frac/cluster_%s' % k[len(prefix):],
                           metrics[k] / metrics['impression'])
    add_summary_fn(
        'cluster_watch_count_frac/no_click',
        metrics['cluster_watch_count_no_click'] / metrics['impression'])
