import React from 'react';
import ReactDOM from 'react-dom/client';
import '@utrecht/component-library-css';
import '@utrecht/design-tokens/dist/index.css';
import '@nl-design-system-candidate/button-react/css';
import '@nl-design-system-candidate/heading-react/css';
import '@nl-design-system-candidate/paragraph-react/css';
import '@nl-design-system-candidate/link-react/css';
import '@nl-design-system-candidate/data-badge-react/css';
import '@nl-design-system-candidate/skip-link-react/css';
import '@nl-design-system-candidate/code-react/css';
import './styles/nlds.css';
import { App } from './App';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
