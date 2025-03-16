import { initializeApp, getApps, getApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: 'AIzaSyDp0du57H9ysUU_W9HwbPJthujFmd9lsA8',
  authDomain: 'financial-qa-chatbot.firebaseapp.com',
  projectId: 'financial-qa-chatbot',
  storageBucket: 'financial-qa-chatbot.firebasestorage.app',
  messagingSenderId: '850831433645',
  appId: '1:850831433645:web:115393152890134f5e4b24',
};

// Initialize Firebase
const app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };
