# CycleGAN-for-cat-eyes

This project focuses on image-to-image translation using Cycle Generative Adversarial Network (CycleGAN). A web interface was also developed using Flask.

The main goal is to input a photo of a person (limited to young men for better results) and generate an image of the same person with cat-like eyes. To achieve this, I used the CelebA dataset and created a custom dataset of positive examples (humans with cat eyes) using DALL·E.

## Notebook and Results

The provided Notebook contains all the code and results of the project.

## How to Execute the Code

To run the code, follow these steps:

1. Download all `.py` files and `.zip` files from this repository.
2. Set up the folder architecture to match the following structure:
   ![Folder Structure](https://github.com/user-attachments/assets/ad88347b-53af-42fe-b48b-b7a0d9d457d6)

3. Open the `app.py` file in any Python IDE of your choice (e.g., VSCode).
4. Update the paths for `UPLOAD_FOLDER` and `checkpoint_path` to match your local setup.

Once the setup is complete, execute the code. A link will be provided in the terminal, which you can open in your browser to use the web interface.

![Web Interface](https://github.com/user-attachments/assets/5d366ee9-bac4-45d2-83b8-ad2b3bf51c97)

## Report

For detailed insights, refer to the report available in the repository: **CycleGAN_report.pdf**.
