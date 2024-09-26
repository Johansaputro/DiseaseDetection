import logo from './logo.svg';
import './App.css';
import './components/NiftiUploader'
import './components/ImageUploader'
import NiftiUploader from './components/NiftiUploader';
import ImageUploader from './components/ImageUploader';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <NiftiUploader />
        <ImageUploader />
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
