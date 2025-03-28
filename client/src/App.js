import './App.css';
import { Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import { Link } from 'react-router-dom';
import Profile from './pages/Profile';
import Login from './pages/Login';
import Register from './pages/Register';
import { Button } from '@mui/material';
import Header from './components/Header';
import { useEffect,useLayoutEffect } from 'react';
import Solution from './pages/Solution';
import { Navigate } from 'react-router-dom';
import ChartPenalty from './pages/ChartPenalty';

function App() {
  
  const MainLayout = ({ children }) => (
    <div>
      <Header />
      <div className='App'>
        <main>{children}</main>
      </div>
    </div>
    
  );


  const NewWindowLayout = ({ children }) => (
    <main>{children}</main>
  );
  

  return (
    <>
      <div>
        <Routes>
          <Route path='/' element={<MainLayout><Home/></MainLayout>}/>
          <Route path='/profile' element={<MainLayout><Profile/></MainLayout>}/>
          <Route path='/login' element={<MainLayout><Login/></MainLayout>}/>
          <Route path='/register' element={<MainLayout><Register/></MainLayout>}/>
          <Route path='/solution/:idSolution' element={<MainLayout><Solution/></MainLayout>}/>
          <Route path='/chart' element={<NewWindowLayout><ChartPenalty/></NewWindowLayout>}/>
        </Routes>
      </div>
    </>
  );
}

export default App;
