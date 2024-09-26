import React, { useState } from 'react';

const NiftiUploader = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = () => {
    const formData = new FormData();
    formData.append('file', file);

    fetch('http://localhost:5000/segmentation/predict', {
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
        console.log(response)
        console.log(response.headers.get('url_list'))
        return response.blob()
      })
      .then(data => {
        // Create a download link for the blob object
        const downloadLink = document.createElement('a');
        downloadLink.href = URL.createObjectURL(data);
        downloadLink.download = 'result.nii.gz';

        // Trigger the download link click
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
      })
      .catch(error => {
        console.error('There was a problem with the file processing:', error);
      });
  };

  return (
    <div>
      <input type="file" accept=".nii.gz" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default NiftiUploader;