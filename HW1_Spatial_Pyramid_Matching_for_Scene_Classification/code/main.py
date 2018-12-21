import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io
import network_layers
import seaborn as sn
import pandas
import os

if __name__ == '__main__':

    num_cores = util.get_num_CPU()

    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"

    image = skimage.io.imread(path_img)
    image = image.astype('float')/255

    # ------------------ Extracts the filter responses for the given image. --------------------
    filter_responses = visual_words.extract_filter_responses(image)
    util.display_filter_responses(filter_responses)

    # ------------- Extracts random samples of the dictionary entries from an image. -----------
    #visual_words.compute_dictionary(num_workers=num_cores)

    # -------------------- Get wordmap given the image and the dictionary. ---------------------
    # dictionary = np.load('dictionary.npy')
    # wordmap = visual_words.get_visual_words(image, dictionary)
    # util.save_wordmap(wordmap, 'tmp.jpg')

    # ------------------- Compute histogram of visual words given the wordmap. -----------------
    # hist = visual_recog.get_feature_from_wordmap(wordmap, dictionary.shape[0])

    # ------------ Compute histogram of visual words using spatial pyramid matching. -----------
    # hist_all = visual_recog.get_feature_from_wordmap_SPM(wordmap, 2, dictionary.shape[0])

    # -------------- Creates a trained recognition system from all training images. ------------
    # visual_recog.build_recognition_system(num_workers=num_cores)

    # ----------------- Evaluate the recognition system using the testing images. --------------
    # conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
    # print(conf)
    # print(accuracy)
    # print(np.diag(conf).sum()/conf.sum())

    # vgg16 = torchvision.models.vgg16(pretrained=True).double()
    # vgg16.eval()
    # vgg16_weights = util.get_VGG16_weights()

    # ----------------- Extract deep features using self-implemented operations. ---------------
    # feat = network_layers.extract_deep_feature(image, vgg16_weights)

    # ---------------- Creates a trained recognition system using deep features. ---------------
    # deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)

    # ----------------------------- Evaluate the recognition system. ---------------------------
    # conf, accuracy = deep_recog.evaluate_recognition_system(vgg16, num_workers=num_cores//2)
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())
    # print(accuracy)

    # ------------------------------ Visualize the confusion matrix. ---------------------------
    # test_data = np.load("../data/test_data.npz")
    # test_image_names = test_data['image_names']
    # test_labels = test_data['labels']
    # label_to_name = set(zip(test_labels, [item[0].split('/')[-2] for item in test_image_names]))
    # columns = [item[1] for item in sorted(list(label_to_name), key=lambda x: x[0])]
    # df_cm = pandas.DataFrame(conf, index=columns, columns=columns)
    # plt.figure(figsize=(10, 7))
    # ax = sn.heatmap(df_cm, cmap='Blues', annot=True)
    # plt.show()



