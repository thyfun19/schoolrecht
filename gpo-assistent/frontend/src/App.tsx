import React from 'react';
import { Document, Surface } from '@utrecht/component-library-react';
import { ZaakvergelijkerPage } from './pages/ZaakvergelijkerPage';

export const App: React.FC = () => {
  return (
    <Document className="utrecht-theme nlds-theme vr-theme">
      <Surface>
        <ZaakvergelijkerPage />
      </Surface>
    </Document>
  );
};

export default App;
