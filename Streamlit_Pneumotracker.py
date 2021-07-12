from Streamlit_Pneumotracker_fonctions import *
from Pneumotracker_func import *

#-----Le streamlit------
def format_func(option):
    return CHOICES[option]

#Génération de la sidebar
status_text = st.sidebar.empty()
add_selectbox = st.sidebar.selectbox(
    "Selectionner un mode",
    ("Détection Pneumonie","Détection Xray","Détails des modèles")
)
#--------------------------------
#Page de détection des pneumonie
#--------------------------------
if add_selectbox == "Détection Pneumonie": # Si l'utilisateur choisi la détection de pneumonie
    st.title("Détection d'une pneumonie dans une radio des poumons d'un enfant")
    st.write(" ")
    CHOICES = {1: "Modèle CNN Simple", 2: "Modèle CNN Complexe", 3: "Modèle VGG16"}
    option = st.selectbox("Choix d'un type de modèle", options=list(CHOICES.keys()), format_func=format_func)

    # --------------------------------
    # Chargement des Modèles de Deep Learning
    # --------------------------------
    model_detection_Xray = keras.models.load_model(
        "./Models/model_detection_Xray.h5")  # Chargement du modèle de détection d'une radio des poumons
    model_segmentation = keras.models.load_model("./Models/unet_lung_seg.hdf5",custom_objects={'dice_coef_loss': dice_coef_loss,
                                                'dice_coef': dice_coef})  # Chargement du modèle de segmentation des poumons
    if option==1:
        model_pneumonie_1_KAGGLE = keras.models.load_model("./Models/model_pneumonie_1_KAGGLE.h5") #Modèle n°1 simple attention au format d'image
        model_pneumonie =model_pneumonie_1_KAGGLE
        model_pneumonie_bis = keras.models.load_model("./Models/model_pneumonie_1_KAGGLE.h5")
        model_pneumonie_bis.layers[-1].activation = None
    elif option==2:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie =model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None

    elif option==3:
        model_pneumonie_3_KAGGLE = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie = model_pneumonie_3_KAGGLE
        model_pneumonie_bis = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie_bis.layers[-1].activation = None

    else:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie = model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None

    # --------------------------------
    # Téléchargement d'une image de radio des poumons
    # --------------------------------
    st.write(f"Vous avez choisi le modèle {option} appelé {format_func(option)}")
    uploaded_file = st.file_uploader("Télécharger une image", accept_multiple_files=False,type=["png", "jpg", "jpeg"])

    if uploaded_file is not None: #Vérification que l'image n'est pas vide
        uploaded_file2 = uploaded_file.read()
        imagePredction = pretraitementImage(uploaded_file2,False,180) #Prétraitements sur l'image
        imagePredction224 = pretraitementImage224(uploaded_file2,False) #Prétraitements sur l'image
        #imagePredction_couleur = pretraitementImage(uploaded_file, True)
        predictions_Xray = predictionModel(imagePredction, model_detection_Xray) #Classification de l'image Xray
        y_pred_Xray = np.round(predictions_Xray).reshape(1, -1)[0]
        #y_pred_Xray = 1
        #predictions_Xray = 1

        if y_pred_Xray == 1: #Si l'image est bien une radio de poumons
            st.write("L'image est une radio des poumons d'un enfant. (_certitude de ",int(predictions_Xray*100),"%_)")
            if option==1:
                predictions_Pneumonie = predictionModel(imagePredction, model_pneumonie)
            else:
                predictions_Pneumonie = predictionModel(imagePredction224, model_pneumonie)
            y_pred = np.round(predictions_Pneumonie).reshape(1, -1)[0]
            labels = ['NORMALE','PNEUMONIA'] #Attention uniquement pour les modèles 2 et 3
            st.write("Le modèle de Deep Learning identifie l'image comme : ","**",labels[int(y_pred)],"**")

            if y_pred == 1:
                 pourcentage_prediction = int(predictions_Pneumonie*100)
            elif y_pred == 0:
                 pourcentage_prediction = (1 - int(predictions_Pneumonie))* 100
            else:
                 pourcentage_prediction = 0
            if pourcentage_prediction <= 95:
                 pourcentage_prediction_text = "Faible"
            elif pourcentage_prediction <= 98 & pourcentage_prediction > 95:
                 pourcentage_prediction_text = "Moyen"
            else:
                 pourcentage_prediction_text = "Elevé"
            st.write("Le pourcentage de certitude est de : ", pourcentage_prediction,"% _(",pourcentage_prediction_text,")_")

            image_segmente = segmentation_image(model_segmentation, uploaded_file2,'','img_segmente.jpeg' )
            st.image(image_segmente)

            #Gram CAM
            if option==1:
                dernierecouche = 'conv2d_2'
            elif option==2:
                dernierecouche = 'conv2d_29'
            elif option==3:
                dernierecouche = 'block5_conv3'
            else:
                dernierecouche = 'conv2d_29'

            #afficherGramCAM(uploaded_file2, model_pneumonie_bis,dernierecouche,False)
            # Gram CAM Segmenté
            #afficherGramCAM(image_segmente, model_pneumonie_seg_bis, 'conv2d_19',True)

            Graphique_Gradcam = get_heatmap_gradcam(model_pneumonie_seg_bis, 'conv2d_169', img_path=None, img=image_segmente, heatmap_quant=None, alpha=0.7)
            st.image(Graphique_Gradcam[0])
            st.image(Graphique_Gradcam[1])

            fig = plt.figure()
            expla = lime_outline(model_pneumonie_seg_bis, img_path=None, img=image_segmente, explanation=None)
            st.pyplot(fig)


            #LIME
            #Travail à faire : réutiliser une varibale déjà en place
            # if option==1:
            #     file_bytes2 = np.asarray(bytearray(uploaded_file2), dtype=np.uint8)
            #     opencv_image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
            #     imageResize = cv2.resize(opencv_image2, (180, 180))
            #     img = skimage.transform.resize(opencv_image2, (180, 180, 1))
            #     img = image.img_to_array(img)
            #     img = np.expand_dims(img, axis=0)
            # else:
            #     file_bytes2 = np.asarray(bytearray(uploaded_file2), dtype=np.uint8)
            #     opencv_image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            #     imageResize = cv2.resize(opencv_image2, (224, 224))
            #     img = skimage.transform.resize(opencv_image2, (224, 224, 3))
            #     img = image.img_to_array(img)
            #     img = np.expand_dims(img, axis=0)
            # #Lancement du modèle LIME
            # explainer = lime_image.LimeImageExplainer()
            # explanation = explainer.explain_instance(img[0].astype('double'),
            #                                          model_pneumonie.predict,
            #                                          top_labels=3,
            #                                          hide_color=0,
            #                                          num_samples=1000)
            # ind = explanation.top_labels[0]
            # dict_heatmap = dict(explanation.local_exp[ind])
            # heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            #
            # #Affichage du LIME
            # fig = plt.figure(figsize=(10,10))
            # plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
            # st.pyplot(fig)
            #
            # #Affichage de l'image avec le LIME
            # fig2 = plt.figure(figsize=(10,10))
            # graph1_1()
            # st.pyplot(fig2)

        else:
            st.write("L'image n'est pas une radio complète des poumons d'un enfant. (_certitude de ",(1 - int(predictions_Xray))* 100,"%_)")

elif add_selectbox == "Détection Xray": #Page de détection de radio des poumons avec une image
    st.title("Détection d'une image de radio des poumons d'enfant")
    st.write(" ")
    model_detection_Xray = keras.models.load_model(".Models/model_detection_Xray.h5")
    uploaded_file = st.file_uploader("Télécharger une image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    if uploaded_file is not None: #Si l'image n'est pas vide
        imagePredction = pretraitementImage(uploaded_file,False)
        predictions_Xray = predictionModel(imagePredction, model_detection_Xray)
        y_pred = np.round(predictions_Xray).reshape(1, -1)[0]
        labels = ['Autre', 'XRAY']
        st.write("Le modèle de Deep Learning identifie l'image comme : ","**",labels[int(y_pred)],"**")
        pourcentage_prediction =1
        if y_pred == 1:
            pourcentage_prediction = int(predictions_Xray*100)
        elif y_pred == 0:
            pourcentage_prediction = (1 - int(predictions_Xray))* 100
        else:
            pourcentage_prediction = 0
        st.write("Le pourcentage de certitude est de : ", pourcentage_prediction)

else:
    CHOICES = {1: "Modèle CNN Simple", 2: "Modèle CNN Complexe", 3: "Modèle VGG16"}
    option = st.selectbox("Choix d'un type de modèle", options=list(CHOICES.keys()), format_func=format_func)
    st.title(f"Détail du modèle de détection de pneumonie {option} appelé {format_func(option)}")
    st.write(" ")
    st.write(" ")
    if option == 1:
        model_pneumonie_1_KAGGLE = keras.models.load_model(
            "./Models/model_pneumonie_1_KAGGLE.h5")  # Modèle n°1 simple attention au format d'image
        model_pneumonie = model_pneumonie_1_KAGGLE
        model_pneumonie_bis = keras.models.load_model("./Models/model_pneumonie_1_KAGGLE.h5")
        model_pneumonie_bis.layers[-1].activation = None
        IMG_SIZE = 180
    elif option == 2:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie = model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None
        IMG_SIZE = 224

    elif option == 3:
        model_pneumonie_3_KAGGLE = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie = model_pneumonie_3_KAGGLE
        model_pneumonie_bis = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie_bis.layers[-1].activation = None
        IMG_SIZE = 224

    else:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie = model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None
        IMG_SIZE=224

    st.write(" ")
    st.write(" ")
    detailModeleUtilise(model_pneumonie,IMG_SIZE)







