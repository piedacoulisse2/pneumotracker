from Streamlit_Pneumotracker_fonctions import *
from Pneumotracker_func import *
import scipy.stats as stats
from Exploration.exploration import *
from Pneumotracker_func import build_df
from Pneumotracker_stats_func import build_stats_df, compare_pops, chi_test



#-----Le streamlit------
def format_func(option):
    return CHOICES[option]
st.set_page_config(
     page_title="Pneumotracker",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )

#Génération de la sidebar
status_text = st.sidebar.empty()


add_selectbox = st.sidebar.selectbox(
    "Selectionner un mode",
    ("Détection Pneumonie","Détails du modèle","Exploration","Statistique")
)
#--------------------------------
#Page de détection des pneumonie
#--------------------------------
if add_selectbox == "Détection Pneumonie": # Si l'utilisateur choisi la détection de pneumonie
    st.title("Détection d'une pneumonie dans une radio des poumons d'un enfant")
    st.write(" ")
    st.subheader("Le module suivant vous permet d'importer une radio d'enfant et de vérifier s'il y a une pneumonie")
    st.write("  ")
    st.write("Le programme va tout d'abord analyser l'image pour savoir si elle contient une radio d'enfant, ensuite elle va classer l'image en Pneumonie probable ou non probable avec un score et enfin elle va indiquer la zone d'analyse sur une image préalablement segmentée.")
    st.write("-----------------------------------------------")
    CHOICES = { 2: "Modèle CNN Complexe", 3: "Modèle VGG16"}
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
    st.write("-----------------------------------------------")
    if uploaded_file is not None: #Vérification que l'image n'est pas vide
        uploaded_file2 = uploaded_file.read()
        imagePredction = pretraitementImage(uploaded_file2,False,180) #Prétraitements sur l'image
        imagePredction224 = pretraitementImage224(uploaded_file2,False) #Prétraitements sur l'image
        #imagePredction_couleur = pretraitementImage(uploaded_file, True)
        predictions_Xray = predictionModel(imagePredction, model_detection_Xray) #Classification de l'image Xray
        y_pred_Xray = np.round(predictions_Xray).reshape(1, -1)[0]
        #y_pred_Xray = 1
        #predictions_Xray = 1
        st.write("  ")
        st.subheader("Analyse de l'image pour déterminer s'il s'agit d'une radio des poumons d'enfant")
        st.write("-----------------------------------------------")

        if y_pred_Xray == 1: #Si l'image est bien une radio de poumons
            st.write("L'image est une radio des poumons d'un enfant. (_certitude de ",int(predictions_Xray*100),"%_)")
            if option==1:
                predictions_Pneumonie = predictionModel(imagePredction, model_pneumonie)
            else:
                predictions_Pneumonie = predictionModel(imagePredction224, model_pneumonie)
            y_pred = np.round(predictions_Pneumonie).reshape(1, -1)[0]
            labels = ['NORMALE','PNEUMONIA'] #Attention uniquement pour les modèles 2 et 3
            st.write("Le modèle de Deep Learning identifie l'image comme : ","**",labels[int(y_pred)],"**")
            st.write("  ")

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

            st.subheader("Segmentation de l'image pour indentifier la zone analysée par le modèle.")
            st.write("-----------------------------------------------")
            image_segmente = segmentation_image(model_segmentation, uploaded_file2,'','img_segmente.jpeg' )
            st.write("La segmentation de l'image se fait grâce à modèle de segentation préentrainé : UNET. L'interet de cette technique est qu'elle permet de ne prendre en compte que les poumons du patient dans la classification de l'image. ")
            st.write("[Plus d'explication du modèle UNET](https://datascientest.com/u-net)")
            st.image(image_segmente,caption="Image segmentée")

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
            st.write("Le GradCam est une technique de visualisation : elle est utile pour comprendre quelles parties d’une image donnée ont conduit au modèle à sa décision finale de classification. Le rendu est sous forme d'une heatmap.")
            st.write(
                "[Plus d'explications de GradCam](https://keras.io/examples/vision/grad_cam/)")
            Graphique_Gradcam = get_heatmap_gradcam(model_pneumonie_seg_bis, 'conv2d_169', img_path=None, img=image_segmente, heatmap_quant=None, alpha=0.7)

            st.image(Graphique_Gradcam[1],caption="GradCam de l'image segmentée")

            st.write("Le modèle Lime (Local Interpretable Model-Agnostic Explanations) permet de donner une pondération d'importance dans le modèle de chaque petit segment de l'image.")
            st.write("[Plus d'explication de LIME](https://github.com/marcotcr/lime)")

            fig = plt.figure()
            expla = lime_outline(model_pneumonie_seg_bis, img_path=None, img=image_segmente, explanation=None)
            st.pyplot(fig)
            st.text("LIME de l'image sélectionnée")



        else:
            st.write("L'image n'est pas une radio complète des poumons. (_certitude de ",(1 - int(predictions_Xray))* 100,"%_)")

elif add_selectbox == "Détection Xray": #Page de détection de radio des poumons avec une image
    st.title("Détection d'une image de radio des poumons d'enfant")
    st.write(" ")
    model_detection_Xray = keras.models.load_model("./Models/model_detection_Xray.h5")
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

elif add_selectbox == "Exploration":
    st.title("Exploration du jeu de données")

    st.subheader("i) Les données sont déjà réparties dans 3 répertoires principaux: test, train et val"
             "Voici la répartition obtenue en nombre de fichiers (5856 au total):"

             )
    st.write(" ")
    df = pd.read_csv("Dfs/export_dataframe.csv")
    #df = pd.read_csv("Dfs/df_all.csv")

    soustotal = df[['Filename', 'Rep']].groupby('Rep').agg('count')
    soustotal['pourcentage'] = round(soustotal / soustotal.sum() * 100, 2)
    st.dataframe(soustotal)

    st.write("On trouve actuellement environ 89 % dans le 'train' et 10.7 % pour train")
    st.write(" ")
    st.subheader("ii) Dans chaque répertoire principal, les données sont aussi classées dans 2 répertoires NORMAL et PNEUMONIA en guise de label")
    st.write(" ")
    st.write("Voici la répartition obtenue globalement")
    st.write(" ")

    st.dataframe(df[['Filename', 'Label_name']].groupby('Label_name').agg('count'))

    st.write("Il y a presque 3 fois plus de cas de pneumonie que de cas normaux. S'agissant de données médicales, cela parait cohérent."
             "A noter: dans les répertoires 'PNEUMONIA', les fichiers comportent les chaines 'virus' ou 'bacteria' "
             "(exemple person88_bacteria_438):")
    st.write(" ")

    pneumonia = df[df['Label_name'] == 'PNEUMONIA']

    pneumo = pneumonia[['Filename', 'Label_pathology']].groupby('Label_pathology').agg('count')

    pneumo['pourcentage'] = round(pneumo / pneumo.sum() * 100, 2)

    st.dataframe(pneumo)

    st.write("On a donc 65% de cas 'bactérie' et 35% de cas 'virus'")

    st.subheader("iii) On peut également faire quelques analyses sur la taille des images en pixels:")
    st.write(" ")
    st.write('Hauteur moyenne :', round(df['Height'].mean(), 2))
    st.write('Hauteur min :', df['Height'].min())
    st.write('Hauteur max :', df['Height'].max())
    st.write()
    st.write('Largeur moyenne :', round(df['Width'].mean(), 2))
    st.write('Largeur min :', df['Width'].min())
    st.write('Largeur max: ', df['Width'].max())

    st.write("Hauteurs comprises entre 127 et 2713, avec une moyenne de 971 pixels environ"
             "Largeurs comprises entre 384 et 2916, avec une moyenne de 1328 pixels environ"
             "On note donc des formats d'image très variables. On a des images présentant des poumons entiers, mais également parfois incomplètes (le haut et le bas des poumons n'apparait pas)"
             "Représentation sous forme de nuage de points:")


    fig1 = plt.figure()
    plt.scatter(df['Height'], df['Width'], s=1)
    plt.scatter(df['Height'].mean(), df['Width'].mean(), color='orange', s=20, label='valeur moyenne')
    plt.xlabel("Hauteur de l'image")
    plt.ylabel("Largeur de l'image")
    plt.legend()
    st.pyplot(fig1)

    st.write("L'image la plus petite mesure 127 x 384 pixels:")
    st.dataframe([df['Size'] == df['Size'].min()])
    st.write("L'image la plus grande mesure 2583 x 2916 pixels:")
    st.write(df[df['Size'] == df['Size'].max()] )
    st.subheader("iv) Images niveaux de gris / couleur / channels")
    st.write("Toutes les images sont au format JPEG.")
    st.dataframe(df[['Filename', 'Mode']].groupby('Mode').agg('count'))
    st.write("avec 'L' (8-bit pixels, black and white) et 'RGB' (3x8-bit pixels, true color)."
             "Les images sont majoritairement en noir et blanc (283 sont en couleur).")
    st.subheader("v) Planche contact")
    list_images = []
    df_normal = df[df['Label_name'] == 'NORMAL']
    for i in df_normal.sample(n=5)['Filepath']:
        list_images.append(i)
    df_pneumonia = df[df['Label_name'] == 'PNEUMONIA']
    for i in df_pneumonia.sample(n=5)['Filepath']:
        list_images.append(i)

    st.write("Affichage au hasard (samples) de 5 images NORMAL et 5 images PNEUMONIA")
    st.image("./Exploration/Streamlit/Exploration_planche_contact.png")

    st.subheader("vi) Histogrammes")
    st.write("On affiche ci-dessous l'histogramme correspondant aux images de la planche contact. Sont également indiquées les valeurs moyennes et médianes.")
    for i in range(1,6):
        st.image("./Exploration/Streamlit/Exploration_histogrammes"+str(i)+".png")


    st.subheader("vii) Analyse des valeurs seuil")
    st.write("La valeur seuil a été fixée à 100.")
    st.write(" ")

    sample_normal = df_normal['Seuil100']

    fig2 = plt.figure()
    plt.hist(sample_normal, bins=300, density=True, label='normal', alpha=0.5)

    sample_pneumonia = df_pneumonia['Seuil100']

    plt.hist(sample_pneumonia, bins=300, density=True, label='pneumonie', alpha=0.5)

    plt.legend();
    st.pyplot(fig2)

    fig3=plt.figure()
    plt.boxplot([sample_normal, sample_pneumonia], labels=['normal', 'pneumo']);
    st.pyplot(fig3)

elif add_selectbox == "Statistique":
    st.title("Statistiques sur le jeu de données")
    orig_path, seg_path = r'.\chest_xray', r'.\segmentation'
    orig_file_ext, seg_file_ext = 'jpeg', 'png'

    st.subheader("I. Building DataFrame for statistics computation")
    print('Building dataset DataFrame')
    print('Building dataset DataFrame')

    df =pd.read_csv(r'.\Dfs\df_all.csv',index_col=0)
    df_stats = pd.read_csv(r'.\Dfs\df_stats.csv',index_col=0)

    st.dataframe(df_stats.head(10))

    st.subheader("II. Mean pixel intensity")

    plot1 = sns.catplot(x='Pathology', y='Mean', data=df_stats.loc[:, ['Mean', 'Pathology']], kind='violin',
                        showmeans=True, inner='box')
    plot1.set_xticklabels(['Normal', 'Pneumonia'], fontsize=15)
    plot1.set_xlabels('')
    plot1.set_ylabels('Mean pixel intensity')
    plt.title('Mean pixel intensity distribution for normal and pneumonia Xray')
    st.pyplot(plot1)

    compare_pops(np.array(df_stats[df_stats['Pathology'] == 'NORMAL']['Mean']),
                 np.array(df_stats[df_stats['Pathology'] == 'PNEUMONIA']['Mean']),
                 'Normal',
                 'Pneumonia')

    #df_chi = df_stats.loc[:, ['Pathology'] + list(np.arange(0, 256, 1))].groupby('Pathology').sum()
    #chi_test(df_chi, 'Pathology', 'Pixel intensity', df_chi.sum().sum())

    st.subheader("III. Pixel intensity threshold")

    plot2 = sns.catplot(x='Pathology', y='Per_thr_100', data=df_stats.loc[:, ['Per_thr_100', 'Pathology']],
                        kind='violin', showmeans=True, inner='box')
    plot2.set_xticklabels(['Normal', 'Pneumonia'], fontsize=15)
    plot2.set_xlabels('')
    plot2.set_ylabels('Mean pixel intensity')
    plt.title('Percentage of pixels with intensity higher than 100')
    st.pyplot(plot2)
    compare_pops(np.array(df_stats[df_stats['Pathology'] == 'NORMAL']['Per_thr_100']),
                 np.array(df_stats[df_stats['Pathology'] == 'PNEUMONIA']['Per_thr_100']),
                 'Normal',
                 'Pneumonia')

    df_chi = df_stats.loc[:, ['Pathology', 'Per_thr_100']]
    df_chi['Per_thr_100'] = df_chi['Per_thr_100'].apply(lambda x: math.trunc(x * 100))
    df_chi = pd.crosstab(df_chi['Pathology'], df_chi['Per_thr_100'])
    chi_test(df_chi, 'Pathology', 'Percentage above 100', df_stats.shape[0])

else:
    CHOICES = { 2: "Modèle CNN Complexe"}
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
        num_model = 1
    elif option == 2:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie = model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None
        IMG_SIZE = 224
        num_model = 2
    elif option == 3:
        model_pneumonie_3_KAGGLE = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie = model_pneumonie_3_KAGGLE
        model_pneumonie_bis = keras.models.load_model("./Models/checkpoint_vgg16_224px_20.h5")
        model_pneumonie_bis.layers[-1].activation = None
        IMG_SIZE = 224
        num_model = 3
    else:
        model_pneumonie_2_KAGGLE = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis = keras.models.load_model("./Models/model_complex_orig_224_rgb.h5")
        model_pneumonie_bis.layers[-1].activation = None
        model_pneumonie = model_pneumonie_2_KAGGLE

        model_pneumonie_seg = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis = keras.models.load_model("./Models/model_complex_seg_224_rgb.h5")
        model_pneumonie_seg_bis.layers[-1].activation = None
        IMG_SIZE=224
        num_model = 2
    st.write(" ")
    st.write(" ")
    detailModeleUtilise(model_pneumonie,IMG_SIZE,num_model)







