# RCNN SAR Flood Detection

Flood detection methods are crucial for developing effective flood mitigation and prevention policies, and are set to become even more important in the decades to come. Building upon the work of R. Liu et al. (2016), Chen et al. (2011) and H. Liu et al. (2024) a modified Random-Coupled Neural Network (RCNN) is here introduced to handle simultaneous noise suppression and propagation of potentially flooded areas combined with an automatic parameter setting method. The model was tested on two datasets from Italy and Germany with F1-scores of approximately 0.38 and 0.35 respectively for the tiled datasets. Testing on the untiled datasets showed a marked improvement with F1-scores reaching approximately 0.63 and 0.60 respectively. Although the results can be characterised as subpar, questions exist about the comparability of the predictions to the ground truths. Regardless of this, serious questions exists about the generalisability of the model, as kernel parameters in the end had to be manually tuned. Further research is needed on a way of setting these parameters.

Primary references: H. Liu et al., “Random-Coupled Neural Network,” 
                    Electronics (Basel), vol. 13, no. 21, pp. 4297-, 2024, 
                    doi: 10.3390/electronics13214297

                    Y. Chen, S. -K. Park, Y. Ma and R. Ala, "A New Automatic Parameter Setting Method of a Simplified PCNN for Image Segmentation," 
                    in IEEE Transactions on Neural Networks, vol. 22, no. 6, pp. 880-892, June 2011, 
                    doi: 10.1109/TNN.2011.2128880

                    R. Liu, Z. Jia, X. Qin, J. Yang, and N. Kasabov, “SAR Image Change Detection Method Based on Pulse-Coupled Neural Network,” 
                    Journal of the Indian Society of Remote Sensing, vol. 44, no. 3, pp. 443–450, 2016, 
                    doi: 10.1007/s12524-015-0507-8
