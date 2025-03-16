import { getApp, getApps, initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: 'AIzaSyDp0du57H9ysUU_W9HwbPJthujFmd9lsA8',
  authDomain: 'financial-qa-chatbot.firebaseapp.com',
  projectId: 'financial-qa-chatbot',
  storageBucket: 'financial-qa-chatbot.firebasestorage.app',
  messagingSenderId: '850831433645',
  appId: '1:850831433645:web:115393152890134f5e4b24',
};
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApp();

// Initialize Firestore
const db = getFirestore(app);

// Note: For client-side Firebase SDK, the database ID is handled by the server-side code
// The client will automatically connect to the database specified in the server configuration

export { db };
