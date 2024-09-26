import React, { useState } from 'react';

const ImageUploader = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://127.0.0.1:5000/segmentation/predict/mrcnn', {
      headers: {
        'Access-Control-Allow-Origin': '*'
      },
      method: 'POST',
      body: formData,
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        // return response.blob();
        return response.json()
      })
      .then(data => {
        console.log(data)
      })
      .catch(error => {
        console.error('There was a problem with the file processing:', error);
      });
  };

  return (
    <div>
      <input type="file" accept=".jpg, .jpeg, .png" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default ImageUploader;