   # Use an official Python runtime as the base image
   FROM python:3.9-slim

   # Set the working directory in the container
   WORKDIR /app

   # Copy the current directory contents into the container
   COPY . /app

   # Install the required packages
   RUN pip install --no-cache-dir -r requirements.txt

   # Make port 8501 available to the world outside this container
   EXPOSE 8501

   # Run streamlit when the container launches
   CMD ["streamlit", "run", "app.py"]