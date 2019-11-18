"""
Performance evaluator

@author Florent Forest
"""

import csv
from somperf.metrics import *
from somperf.utils.topology import rectangular_topology_dist
import sklearn.metrics as skmetrics


class PerfLogger:

    def __init__(self,
                 with_validation=False,
                 with_labels=False,
                 with_latent_metrics=False,
                 save_dir='results/tmp'):
        print('Initializing PerfLogger.')
        self.with_validation = with_validation

        # Metrics monitored during training
        self.metrics = [
            'iteration',
            'T',
            'L',
            'Lr',
            'Lsom',
            'quantization_error',
            'topographic_error',
            'combined_error',
            'silhouette'
        ]

        # Metrics evaluated on entire dataset after training
        self.evaluation_metrics = [
            'iteration',
            # 'combined_error',
            # 'kruskal_shepard_error',
            # 'neighborhood_preservation',
            # 'trustworthiness',
            # 'quantization_error',
            # 'topographic_error',
            # 'silhouette',
            'combined_error_val',
            'kruskal_shepard_error_val',
            'neighborhood_preservation_val',
            'trustworthiness_val',
            'quantization_error_val',
            'topographic_error_val',
            'silhouette_val',
            'topographic_product'
        ]

        if with_labels:
            self.metrics += [
                # 'accuracy',
                'purity',
                'nmi',
                'ari'
            ]
            self.evaluation_metrics += [
                # 'accuracy',
                'purity_val',
                'nmi_val',
                'ari_val',
                'class_scatter_index_val',
                'entropy_val'
            ]

        if with_latent_metrics:
            self.metrics += [
                'latent_quantization_error',
                'latent_topographic_error',
                'latent_combined_error',
                'latent_silhouette'
            ]
            self.evaluation_metrics += [
                # 'latent_combined_error',
                # 'latent_kruskal_shepard_error',
                # 'latent_neighborhood_preservation',
                # 'latent_trustworthiness',
                # 'latent_quantization_error',
                # 'latent_topographic_error',
                # 'latent_silhouette',
                'latent_combined_error_val',
                'latent_kruskal_shepard_error_val',
                'latent_neighborhood_preservation_val',
                'latent_trustworthiness_val',
                'latent_quantization_error_val',
                'latent_topographic_error_val',
                'latent_silhouette_val',
                'latent_topographic_product'
            ]

        if with_validation:
            self.metrics += [metric + '_val' for metric in self.metrics if metric not in ['iteration',
                                                                                          'T',
                                                                                          'topographic_product',
                                                                                          'latent_topographic_product']]

        self.logfile = open(save_dir + '/log.csv', 'w')
        self.logwriter = csv.DictWriter(self.logfile, self.metrics)
        self.logwriter.writeheader()

        self.evalfile = open(save_dir + '/evaluation.csv', 'w')
        self.evalwriter = csv.DictWriter(self.evalfile, self.evaluation_metrics)
        self.evalwriter.writeheader()

    def __delete__(self):
        self.close()

    def close(self):
        print('Closing PerfLogger.')
        self.logfile.close()

    def log(self, summary, verbose=0):
        """Log monitored metrics.

        Parameters
        ----------
        summary : dict
            training summary
        verbose : int
            verbosity level
            0 = print nothing
            1 = print only iteration number and losses
            2 = print all monitored metrics
        """

        results = self._compute_metrics(summary, self.metrics, verbose=False)

        if verbose > 0:
            print('iteration {} - T={}'.format(results['iteration'], results['T']))
            if verbose == 1:
                print('[Train] - Lr={:f}, Lsom={:f}, L={:f}'.format(results['Lr'], results['Lsom'], results['L']))
                if self.with_validation:
                    print('[Val] - Lr={:f}, Lsom={:f}, L={:f}'.format(results['Lr_val'], results['Lsom_val'],
                                                                       results['L_val']))
            if verbose >= 2:
                print(', '.join(['{}={:f}'.format(metric, results[metric]) for metric in self.metrics]))

        self.logwriter.writerow(results)

    def evaluate(self, summary, verbose=0):
        """Save evaluation metrics.

        Parameters
        ----------
        summary : dict
            training summary
        verbose : int
            verbosity level
            0 = print nothing
            1 = print all evaluated metrics
        """

        results = self._compute_metrics(summary, self.evaluation_metrics, verbose=True)

        if verbose > 0:
            print(', '.join(['{}={:f}'.format(metric, results[metric]) for metric in self.evaluation_metrics]))

        self.evalwriter.writerow(results)

    @staticmethod
    def _compute_metrics(summary, metrics, verbose=False):
        """Computes selected metrics from a training summary.

        Parameters
        ----------
        summary : dict
            training summary
        metrics : list
            list of metrics to compute
        verbose : boolean
            print metric being computed

        Returns
        -------
        results : dict
            metrics
        """
        results = {}

        # Basic info
        if 'iteration' in metrics:
            results['iteration'] = summary['iteration']
        if 'T' in metrics:
            results['T'] = summary['T']

        # Losses
        if 'L' in metrics:
            results['L'] = summary['loss'][0]
        if 'Lr' in metrics:
            results['Lr'] = summary['loss'][1]
        if 'Lsom' in metrics:
            results['Lsom'] = summary['loss'][2]
        if 'L_val' in metrics:
            results['L_val'] = summary['val_loss'][0]
        if 'Lr_val' in metrics:
            results['Lr_val'] = summary['val_loss'][1]
        if 'Lsom_val' in metrics:
            results['Lsom_val'] = summary['val_loss'][2]

        # Internal indices
        dist_fun = rectangular_topology_dist(summary['map_size'])

        # Combined error
        if 'combined_error' in metrics:
            if verbose:
                print('Evaluating combined_error...')
            results['combined_error'] = combined_error(dist_fun, som=summary['prototypes'], d=summary['d_original'])
        if 'latent_combined_error' in metrics:
            if verbose:
                print('Evaluating latent_combined_error...')
            results['latent_combined_error'] = combined_error(dist_fun, som=summary['latent_prototypes'],
                                                              d=summary['d_latent'])
        if 'combined_error_val' in metrics:
            if verbose:
                print('Evaluating combined_error_val...')
            results['combined_error_val'] = combined_error(dist_fun, som=summary['prototypes'],
                                                           d=summary['d_original_val'])
        if 'latent_combined_error_val' in metrics:
            if verbose:
                print('Evaluating latent_combined_error_val...')
            results['latent_combined_error_val'] = combined_error(dist_fun, som=summary['latent_prototypes'],
                                                                  d=summary['d_latent_val'])

        # Kruskal-Shepard error
        if 'kruskal_shepard_error' in metrics:
            if verbose:
                print('Evaluating kruskal_shepard_error...')
            results['kruskal_shepard_error'] = kruskal_shepard_error(dist_fun, x=summary['X'], d=summary['d_original'])
        if 'latent_kruskal_shepard_error' in metrics:
            if verbose:
                print('Evaluating latent_kruskal_shepard_error...')
            results['latent_kruskal_shepard_error'] = kruskal_shepard_error(dist_fun, x=summary['Z'],
                                                                            d=summary['d_latent'])
        if 'kruskal_shepard_error_val' in metrics:
            if verbose:
                print('Evaluating kruskal_shepard_error_val...')
            results['kruskal_shepard_error_val'] = kruskal_shepard_error(dist_fun, x=summary['X_val'],
                                                                         d=summary['d_original_val'])
        if 'latent_kruskal_shepard_error_val' in metrics:
            if verbose:
                print('Evaluating latent_kruskal_shepard_error_val...')
            results['latent_kruskal_shepard_error_val'] = kruskal_shepard_error(dist_fun, x=summary['Z_val'],
                                                                                d=summary['d_latent_val'])

        # Neighborhood preservation & Trustworthiness
        if 'neighborhood_preservation' in metrics or 'trustworthiness' in metrics:
            if verbose:
                print('Evaluating neighborhood_preservation_trustworthiness...')
            npr, tr = neighborhood_preservation_trustworthiness(1, som=summary['prototypes'], x=summary['X'],
                                                                d=summary['d_original'])
            if 'neighborhood_preservation' in metrics:
                results['neighborhood_preservation'] = npr
            if 'trustworthiness' in metrics:
                results['trustworthiness'] = tr
        if 'latent_neighborhood_preservation' in metrics or 'latent_trustworthiness' in metrics:
            if verbose:
                print('Evaluating latent_neighborhood_preservation_trustworthiness...')
            npr, tr = neighborhood_preservation_trustworthiness(1, som=summary['latent_prototypes'], x=summary['Z'],
                                                                d=summary['d_latent'])
            if 'latent_neighborhood_preservation' in metrics:
                results['latent_neighborhood_preservation'] = npr
            if 'latent_trustworthiness' in metrics:
                results['latent_trustworthiness'] = tr
        if 'neighborhood_preservation_val' in metrics or 'trustworthiness_val' in metrics:
            if verbose:
                print('Evaluating neighborhood_preservation_trustworthiness_val...')
            npr, tr = neighborhood_preservation_trustworthiness(1, som=summary['prototypes'],
                                                                x=summary['X_val'], d=summary['d_original_val'])
            if 'neighborhood_preservation_val' in metrics:
                results['neighborhood_preservation_val'] = npr
            if 'trustworthiness_val' in metrics:
                results['trustworthiness_val'] = tr
        if 'latent_neighborhood_preservation_val' in metrics or 'latent_trustworthiness_val' in metrics:
            print('Evaluating latent_neighborhood_preservation_trustworthiness_val...')
            npr, tr = neighborhood_preservation_trustworthiness(1, som=summary['latent_prototypes'], x=summary['Z_val'],
                                                                d=summary['d_latent_val'])
            if 'latent_neighborhood_preservation_val' in metrics:
                results['latent_neighborhood_preservation_val'] = npr
            if 'latent_trustworthiness_val' in metrics:
                results['latent_trustworthiness_val'] = tr

        # Quantization error
        if 'quantization_error' in metrics:
            if verbose:
                print('Evaluating quantization_error...')
            results['quantization_error'] = quantization_error(d=summary['d_original'])
        if 'latent_quantization_error' in metrics:
            if verbose:
                print('Evaluating quantization_error...')
            results['latent_quantization_error'] = quantization_error(d=summary['d_latent'])
        if 'quantization_error_val' in metrics:
            if verbose:
                print('Evaluating quantization_error_val...')
            results['quantization_error_val'] = quantization_error(d=summary['d_original_val'])
        if 'latent_quantization_error_val' in metrics:
            if verbose:
                print('Evaluating latent_quantization_error_val...')
            results['latent_quantization_error_val'] = quantization_error(d=summary['d_latent_val'])

        # Topographic error
        if 'topographic_error' in metrics:
            if verbose:
                print('Evaluating topographic_error...')
            results['topographic_error'] = topographic_error(dist_fun, d=summary['d_original'])
        if 'latent_topographic_error' in metrics:
            if verbose:
                print('Evaluating latent_topographic_error...')
            results['latent_topographic_error'] = topographic_error(dist_fun, d=summary['d_latent'])
        if 'topographic_error_val' in metrics:
            if verbose:
                print('Evaluating topographic_error_val...')
            results['topographic_error_val'] = topographic_error(dist_fun, d=summary['d_original_val'])
        if 'latent_topographic_error_val' in metrics:
            if verbose:
                print('Evaluating latent_topographic_error_val...')
            results['latent_topographic_error_val'] = topographic_error(dist_fun, d=summary['d_latent_val'])

        # Topographic product
        if 'topographic_product' in metrics:
            if verbose:
                print('Evaluating topographic_product...')
            results['topographic_product'] = topographic_product(dist_fun, som=summary['prototypes'])
        if 'latent_topographic_product' in metrics:
            if verbose:
                print('Evaluating latent_topographic_product...')
            results['latent_topographic_product'] = topographic_product(dist_fun, som=summary['latent_prototypes'])

        # Silhouette
        if 'silhouette' in metrics:
            if verbose:
                print('Evaluating silhouette...')
            results['silhouette'] = skmetrics.silhouette_score(summary['X'], summary['y_pred'])
        if 'latent_silhouette' in metrics:
            if verbose:
                print('Evaluating latent_silhouette...')
            results['latent_silhouette'] = skmetrics.silhouette_score(summary['Z'], summary['y_pred'])
        if 'silhouette_val' in metrics:
            if verbose:
                print('Evaluating silhouette_val...')
            results['silhouette_val'] = skmetrics.silhouette_score(summary['X_val'], summary['y_val_pred'])
        if 'latent_silhouette_val' in metrics:
            if verbose:
                print('Evaluating latent_silhouette_val...')
            results['latent_silhouette_val'] = skmetrics.silhouette_score(summary['Z_val'], summary['y_val_pred'])

        # External indices

        # Clustering accuracy
        if 'accuracy' in metrics:
            if verbose:
                print('Evaluating accuracy...')
            results['accuracy'] = clustering_accuracy(summary['y_true'], summary['y_pred'])
        if 'accuracy_val' in metrics:
            if verbose:
                print('Evaluating accuracy_val...')
            results['accuracy_val'] = clustering_accuracy(summary['y_val_true'], summary['y_val_pred'])

        # Purity
        if 'purity' in metrics:
            if verbose:
                print('Evaluating purity...')
            results['purity'] = purity(summary['y_true'], summary['y_pred'])
        if 'purity_val' in metrics:
            if verbose:
                print('Evaluating purity_val...')
            results['purity_val'] = purity(summary['y_val_true'], summary['y_val_pred'])

        # NMI
        if 'nmi' in metrics:
            if verbose:
                print('Evaluating nmi...')
            results['nmi'] = skmetrics.normalized_mutual_info_score(summary['y_true'], summary['y_pred'])
        if 'nmi_val' in metrics:
            if verbose:
                print('Evaluating nmi_val...')
            results['nmi_val'] = skmetrics.normalized_mutual_info_score(summary['y_val_true'], summary['y_val_pred'])

        # ARI
        if 'ari' in metrics:
            if verbose:
                print('Evaluating ari...')
            results['ari'] = skmetrics.adjusted_rand_score(summary['y_true'], summary['y_pred'])
        if 'ari_val' in metrics:
            if verbose:
                print('Evaluating ari_val...')
            results['ari_val'] = skmetrics.adjusted_rand_score(summary['y_val_true'], summary['y_val_pred'])

        # Class scatter index
        if 'class_scatter_index' in metrics:
            if verbose:
                print('Evaluating csi...')
            results['class_scatter_index'] = class_scatter_index(dist_fun, summary['y_true'], summary['y_pred'])
        if 'class_scatter_index_val' in metrics:
            if verbose:
                print('Evaluating csi_val...')
            results['class_scatter_index_val'] = class_scatter_index(dist_fun, summary['y_val_true'],
                                                                     summary['y_val_pred'])

        # Entropy
        if 'entropy' in metrics:
            if verbose:
                print('Evaluating entropy...')
            results['entropy'] = entropy(summary['y_true'], summary['y_pred'])
        if 'entropy_val' in metrics:
            if verbose:
                print('Evaluating entropy_val...')
            results['entropy_val'] = entropy(summary['y_val_true'], summary['y_val_pred'])
        
        return results
