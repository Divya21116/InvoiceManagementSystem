# import librosa
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# import io
# from tensorflow import keras
# import tempfile
# from transformers import pipeline
# from keras.preprocessing.image import load_img, img_to_array
# from PIL import Image
# from io import BytesIO
# from django.core.files.uploadedfile import InMemoryUploadedFile
# #load the model

# #define the function to detect the audio
# def detect_fake(filename):
#     model_path = r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\model_generated1.h5'
#     model=load_model(model_path)
#     sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
#     mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
#     mfccs_features_scaled = np.mean(mfcc_features.T, axis=0)
#     mfccs_features_scaled = mfccs_features_scaled.reshape(1, -1)
#     result_array = model.predict(mfccs_features_scaled)
#     confidence = result_array* 100
#     print(confidence)
#     # print(result_array)
#     result_classes = ["Fake Audio", "Real Audio"]
#     result = np.argmax(result_array[0])
#     return result_classes[result]

# # # Create a function to open, crop and resize images
# # def load_and_preprocess_real_images(image_path, target_size=(64, 64)):
# #     # Open the image
# #     img = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure 3 channels
# #     # Crop 20 pixels from the top and bottom to make it square
# #     img = img.crop((0, 20, 178, 198))
# #     # Resize the image
# #     img = img.resize(target_size)
# #     # Convert to numpy array and scale to [-1, 1]
# #     img = np.array(img)/127.5 - 1
# #     return img
# # def detect_image(image_file):
# #     # Load the pre-trained GAN model
# #     gan_model = keras.models.load_model(r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\xception_deepfake_image_100.h5')

# #     # Load and preprocess the image
# #     img_array = load_and_preprocess_real_images(image_file)
# #     img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, 64, 64, 3)

# #     # Predict using the discriminator component of the GAN model
# #     predictions = gan_model.layers[1](img_array)

# #     # Get the probabilities
# #     real_prob = predictions[0][0]
# #     fake_prob = predictions[0][1]  # Assuming the model returns real and fake probabilities separately

# #     # Convert probabilities to NumPy arrays
# #     real_prob_np = real_prob.numpy()
# #     fake_prob_np = fake_prob.numpy()

# #     # Set a threshold for classification
# #     THRESHOLD = 0.5  # Adjust based on your needs

# #     # Make the prediction
# #     # Make the prediction
# #     if real_prob_np.max() > fake_prob_np.max():
# #         return "Real Image"
# #     elif fake_prob_np.max() > real_prob_np.max():
# #         return "Fake Image"
# #     else:
# #         return "Uncertain"  # Both probabilities are too close
# import cv2
# import numpy as np
# def read_image_file(file):
#     # Convert InMemoryUploadedFile to a byte array
#     image_stream = io.BytesIO(file.read())
#     image = Image.open(image_stream)
#     # Convert the PIL image to a NumPy array (which OpenCV expects)
#     return np.array(image)
# def resize_aspect_ratio(image, target_size=(224, 224)):
#     h, w = image.shape[:2]
#     # Compute aspect ratio and resize accordingly
#     if w > h:
#         new_w = target_size[1]
#         new_h = int(h * new_w / w)
#     else:
#         new_h = target_size[0]
#         new_w = int(w * new_h / h)
    
#     resized_image = cv2.resize(image, (new_w, new_h))
#     return resized_image

# def detect_image(image_file):
#     # image_file=r'C:\Users\divya.gugulothu\Downloads\deep_fake_detection_image_CNN_xceptionNet\artifacts\aaeucwtkdx.jpg'
#     # img_array = read_image_file(image_file)
#     img_array = cv2.imread(image_file)
#     # Resize image to (224, 224, 3)
#     resized_image = resize_aspect_ratio(img_array, target_size=(224, 224))
#     print(resized_image.shape)
#     # Reshape to (1, 224, 224, 3)
#     resized_image = cv2.resize(resized_image, (224, 224))
#     #reshaped_image_array = resized_image.reshape(1,224,224,3)
#     #print(reshaped_image_array.shape)
#     x = np.expand_dims(resized_image, axis=0)
#     x = x/255
#     model = load_model(r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\xception_deepfake_image_model.keras')
#     # model.summary()
#     prediction = model.predict(x)[0][0]
    
#     print(prediction)    
    
#     if prediction > 0.5: ###### deepfake is labelled as 1 in this model training
#         print('Image is a deepfake image')
#         return 'Fake Image'
#     else:
#         print('Image is a Real image')
#         return 'Real Image'
# #  model = load_model(r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\xception_deepfake_image_model.keras')
# IMG_SIZE = 224
# MAX_SEQ_LENGTH = 20
# NUM_FEATURES = 2048
# def crop_center_square(frame):
#     y, x = frame.shape[0:2]
#     min_dim = min(y, x)
#     start_x = (x // 2) - (min_dim // 2)
#     start_y = (y // 2) - (min_dim // 2)
#     return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


# def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
#     cap = cv2.VideoCapture(path)
#     frames = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = crop_center_square(frame)
#             frame = cv2.resize(frame, resize)
#             frame = frame[:, :, [2, 1, 0]]
#             frames.append(frame)

#             if len(frames) == max_frames:
#                 break
#     finally:
#         cap.release()
#     return np.array(frames)
# def build_feature_extractor():
#     feature_extractor = keras.applications.InceptionV3(
#         weights="imagenet",
#         include_top=False,
#         pooling="avg",
#         input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     )
#     preprocess_input = keras.applications.inception_v3.preprocess_input

#     inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
#     preprocessed = preprocess_input(inputs)

#     outputs = feature_extractor(preprocessed)
#     return keras.Model(inputs, outputs, name="feature_extractor")


# feature_extractor = build_feature_extractor()

# def sequence_prediction(path,model):
#     frames = load_video(path)
#     frame_features, frame_mask = prepare_single_video(frames)
#     print(frame_features)
#     return model.predict([frame_features, frame_mask])[0]
# def prepare_single_video(frames):
#     frames = frames[None, ...]
#     frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
#     frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

#     for i, batch in enumerate(frames):
#         video_length = batch.shape[0]
#         length = min(MAX_SEQ_LENGTH, video_length)
#         for j in range(length):
#             frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
#         frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

#     return frame_features, frame_mask

# # def to_gif(images):
# #     converted_images = images.astype(np.uint8)
# #     imageio.mimsave("animation.gif", converted_images, fps=10)
# #     return embed.embed_file("animation.gif")

# def detect_video(video_file):
#     # frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
#     # mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
#     # #mask_input = tf.sequence_mask(sequences_length, maxlen=MAX_SEQ_LENGTH)  # Right-padded mask
#     # # Refer to the following tutorial to understand the significance of using `mask`:
#     # # https://keras.io/api/layers/recurrent_layers/gru/
#     # x = keras.layers.GRU(16, return_sequences=True, use_cudnn=False)(
#     #     frame_features_input, mask=mask_input,
#     # )
#     # x = keras.layers.GRU(8)(x)
#     # x = keras.layers.Dropout(0.4)(x)
#     # x = keras.layers.Dense(8, activation="relu")(x)
#     # output = keras.layers.Dense(1, activation="sigmoid")(x)

#     # model = keras.Model([frame_features_input, mask_input], output)
#     # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     # model.summary()
#     # model_path = r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\checkpoint_100.weights.h5'
#     # model.load_weights(model_path)
#     # frames = load_video(video_file)
#     model = tf.keras.models.load_model(r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\models\checkpoint_177.model.keras')
#     # model.summary()
#     prediction = sequence_prediction(video_file,model)
#     confidence = prediction *100
#     if prediction >=0.5:
#         print(f'The predicted class of the video is FAKE with cinfidence of ', confidence)
#         return 'Fake Video'
#     else:
#         print(f'The predicted class of the video is REAL with cinfidence of ', confidence)
#         return 'Real Video'
# def detect_text(text):
#     BERT_MODEL = "distilbert-base-cased"
#     output_dir = r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\ai-generated-essay-detection-distilbert'
#     # Your logic to process and detect fake text content
#     pipe = pipeline("text-classification", model = output_dir, tokenizer=BERT_MODEL)
#     result = "Fake Text Detected"  # Example result
#     # Get the result from the pipeline
#     prediction = pipe(text, top_k=10)[0]  # Take the top prediction
#     # Extract only the label
#     result = prediction['label']
#     return result
# from reportlab.lib import colors
# from reportlab.lib.pagesizes import letter, A4
# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from io import BytesIO

# def number_to_words(n):
#     """Convert number to words (Indian format)"""
#     ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
#     tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
#     teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    
#     def convert_hundreds(num):
#         result = ''
#         if num >= 100:
#             result += ones[num // 100] + ' hundred '
#             num %= 100
#         if num >= 20:
#             result += tens[num // 10] + ' '
#             num %= 10
#         elif num >= 10:
#             result += teens[num - 10] + ' '
#             return result
#         if num > 0:
#             result += ones[num] + ' '
#         return result

#     if n == 0:
#         return 'zero'
    
#     result = ''
#     crores = n // 10000000
#     if crores > 0:
#         result += convert_hundreds(crores) + 'crore '
#         n %= 10000000
    
#     lakhs = n // 100000
#     if lakhs > 0:
#         result += convert_hundreds(lakhs) + 'lakh '
#         n %= 100000
    
#     thousands = n // 1000
#     if thousands > 0:
#         result += convert_hundreds(thousands) + 'thousand '
#         n %= 1000
    
#     if n > 0:
#         result += convert_hundreds(n)
    
#     return result.strip() + ' rupees only'

# def generate_pdf(invoice):
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
#     # Container for the 'Flowable' objects
#     elements = []
    
#     # Define styles
#     styles = getSampleStyleSheet()
#     title_style = ParagraphStyle(
#         'CustomTitle',
#         parent=styles['Heading1'],
#         fontSize=16,
#         textColor=colors.black,
#         spaceAfter=12,
#         alignment=1  # Center alignment
#     )
    
#     # Company Header
#     company_data = [
#         [Paragraph(f"<b>{invoice.company.name}</b>", title_style)],
#         [Paragraph(invoice.company.subtitle, styles['Normal'])],
#         [Paragraph(f"{invoice.company.address}", styles['Normal'])],
#         [Paragraph(f"{invoice.company.city}, {invoice.company.state} - {invoice.company.pincode}", styles['Normal'])],
#         [Paragraph(f"PH: {invoice.company.phone}", styles['Normal'])],
#         [Paragraph(f"<b>TAX {invoice.document_type.upper()}</b>", title_style)],
#     ]
    
#     company_table = Table(company_data, colWidths=[6*inch])
#     company_table.setStyle(TableStyle([
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
#     ]))
#     elements.append(company_table)
#     elements.append(Spacer(1, 12))
    
#     # Invoice details and customer info
#     info_data = [
#         [f"GSTIN NO: {invoice.company.gstin}", f"Transportation Mode: {invoice.transport_mode}"],
#         [f"{invoice.document_type.title()} S.No: {invoice.invoice_number}", f"Vehicle No: {invoice.vehicle_number}"],
#         [f"Date: {invoice.date}", f"Place Of Supply: {invoice.place_of_supply}"],
#     ]
    
#     info_table = Table(info_data, colWidths=[3*inch, 3*inch])
#     info_table.setStyle(TableStyle([
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#     ]))
#     elements.append(info_table)
#     elements.append(Spacer(1, 12))
    
#     # Customer details
#     customer_data = [
#         ["Details Of Receiver (Billed to)", "Details Of Consignee (Shipped to)"],
#         [f"Name: {invoice.customer.name}", f"Name: {invoice.customer.name}"],
#         [f"Address: {invoice.customer.address}", f"Address: {invoice.customer.address}"],
#         [f"GSTIN NO: {invoice.customer.gstin}", f"GSTIN NO: {invoice.customer.gstin}"],
#     ]
    
#     customer_table = Table(customer_data, colWidths=[3*inch, 3*inch])
#     customer_table.setStyle(TableStyle([
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#     ]))
#     elements.append(customer_table)
#     elements.append(Spacer(1, 12))
    
#     # Items table
#     items_data = [['Sl.No.', 'HSN CODE', 'Description', 'W', 'H', 'Qty', 'Sft', 'Rate', 'Amount']]
    
#     for idx, item in enumerate(invoice.items.all(), 1):
#         sft = float(item.width) * float(item.height)
#         items_data.append([
#             str(idx),
#             item.hsn_code,
#             item.description,
#             str(item.width),
#             str(item.height),
#             str(item.quantity),
#             str(sft),
#             str(item.rate),
#             f"₹{item.amount:.2f}"
#         ])
    
#     if invoice.transportation_expenses > 0:
#         items_data.append(['', '', 'Transportation expenses', '', '', '', '', '', f"₹{invoice.transportation_expenses:.2f}"])
    
#     items_table = Table(items_data, colWidths=[0.5*inch, 0.8*inch, 2*inch, 0.4*inch, 0.4*inch, 0.4*inch, 0.5*inch, 0.6*inch, 0.8*inch])
#     items_table.setStyle(TableStyle([
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
#         ('FONTSIZE', (0, 0), (-1, -1), 9),
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
#     ]))
#     elements.append(items_table)
#     elements.append(Spacer(1, 12))
    
#     # Footer with totals and bank details
#     footer_left = [
#         f"In Words: {number_to_words(int(invoice.grand_total))}",
#         "",
#         f"Bank Details: {invoice.company.name}",
#         f"Account Number: {invoice.company.account_number}",
#         f"Branch: {invoice.company.branch}",
#         f"IFSC Code: {invoice.company.ifsc}",
#     ]
    
#     footer_right = [
#         ["SUBTOTAL", f"₹{invoice.subtotal:.2f}"],
#         [f"CGST + SGST {invoice.cgst_sgst_rate}%", f"₹{invoice.tax_amount:.2f}"],
#         ["GRAND TOTAL", f"₹{invoice.grand_total:.2f}"],
#         ["", ""],
#         ["Authorised Signatory", ""],
#         [f"Name: {invoice.company.authorized_signatory}", ""],
#     ]
    
#     footer_table = Table([
#         ['\n'.join(footer_left), Table(footer_right, colWidths=[1.8*inch, 1*inch])]
#     ], colWidths=[3*inch, 3*inch])
    
#     footer_table.setStyle(TableStyle([
#         ('BOX', (0, 0), (-1, -1), 1, colors.black),
#         ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('VALIGN', (0, 0), (-1, -1), 'TOP'),
#         ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
#     ]))
    
#     # Style the inner table
#     inner_table = footer_table._cellvalues[0][1]
#     inner_table.setStyle(TableStyle([
#         ('FONTSIZE', (0, 0), (-1, -1), 9),
#         ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
#         ('LINEBELOW', (0, -2), (-1, -2), 1, colors.black),
#         ('FONTNAME', (0, -2), (-1, -2), 'Helvetica-Bold'),
#     ]))
    
#     elements.append(footer_table)
    
#     # Build PDF
#     doc.build(elements)
#     pdf_content = buffer.getvalue()
#     buffer.close()
    
#     return pdf_content
