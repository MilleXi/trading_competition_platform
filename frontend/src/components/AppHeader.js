import React, { useEffect, useRef } from 'react'
import { NavLink } from 'react-router-dom'
import { useSelector, useDispatch } from 'react-redux'
import {
  CContainer,
  CDropdown,
  CDropdownItem,
  CDropdownMenu,
  CDropdownToggle,
  CHeader,
  CHeaderNav,
  CHeaderToggler,
  CNavLink,
  CNavItem,
  useColorModes,
} from '@coreui/react'
import CIcon from '@coreui/icons-react'
import {
  cilBell,
  cilContrast,
  cilEnvelopeOpen,
  cilList,
  cilMenu,
  cilMoon,
  cilSun,
} from '@coreui/icons'

import { AppBreadcrumb } from './index'
import { AppHeaderDropdown } from './header/index'
import { hexToRgba } from '@coreui/utils'

const AppHeader = () => {

  const headerRef = useRef()
  const { colorMode, setColorMode } = useColorModes('coreui-free-react-admin-template-theme')

  const dispatch = useDispatch()
  const sidebarShow = useSelector((state) => state.sidebarShow)

  useEffect(() => {
    document.addEventListener('scroll', () => {
      headerRef.current &&
        headerRef.current.classList.toggle('shadow-sm', document.documentElement.scrollTop > 0)
    })
  }, [])

  // console.log('Header rendered');
  return (
    <CHeader position="sticky" className="mb-4 p-0" ref={headerRef} style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)', borderBottom: 'none' }}>
      <CContainer className="px-4" fluid>
        {/* 左上角打开菜单栏的按钮，删除
        <CHeaderToggler
          onClick={() => dispatch({ type: 'set', sidebarShow: !sidebarShow })}
          style={{ marginInlineStart: '-14px' }}
        >
          <CIcon icon={cilMenu} size="lg" />
        </CHeaderToggler> */}
        <CHeaderNav className="d-none d-md-flex">
          <CNavItem>
            <CNavLink to="/home" as={NavLink}>
              <div style={{ color: 'white' }}>Home</div>
            </CNavLink>
          </CNavItem>
          {/* Users按钮删除
          <CNavItem>
            <CNavLink href="/#/users">Users</CNavLink>
          </CNavItem> */}


        </CHeaderNav>
        <CHeaderNav className="ms-auto">
          {/* 删除右侧原本的按钮
          <CNavItem>
            <CNavLink href="#">
              <CIcon icon={cilBell} size="lg" />
            </CNavLink>
          </CNavItem>
          <CNavItem>
            <CNavLink href="#">
              <CIcon icon={cilList} size="lg" />
            </CNavLink>
          </CNavItem>
          <CNavItem>
            <CNavLink href="#">
              <CIcon icon={cilEnvelopeOpen} size="lg" />
            </CNavLink>
          </CNavItem> */}
          {/* 添加设置的下拉菜单 */}
          <CDropdown variant="nav-item">
            <CDropdownToggle caret={true}>
              <span style={{ color: 'white' }} >Settings</span>
            </CDropdownToggle>

            <CDropdownMenu>
              <CDropdownItem>
                <span style={{ color: 'white' }} >Theme</span>
              </CDropdownItem>

              <CDropdownItem>
                <span style={{ color: 'white' }} >Language</span>
              </CDropdownItem>

              <CDropdownItem>
                <span style={{ color: 'white' }} >Account</span>
              </CDropdownItem>

            </CDropdownMenu>
          </CDropdown>


        </CHeaderNav>

        {/* <CHeaderNav> */}
        {/* 删除亮度调节按钮
          <li className="nav-item py-1">
            <div className="vr h-100 mx-2 text-body text-opacity-75"></div>
          </li>
          <CDropdown variant="nav-item" placement="bottom-end">
            <CDropdownToggle caret={false}>
              {colorMode === 'dark' ? (
                <CIcon icon={cilMoon} size="lg" />
              ) : colorMode === 'auto' ? (
                <CIcon icon={cilContrast} size="lg" />
              ) : (
                <CIcon icon={cilSun} size="lg" />
              )}
            </CDropdownToggle>
            <CDropdownMenu>
              <CDropdownItem
                active={colorMode === 'light'}
                className="d-flex align-items-center"
                as="button"
                type="button"
                onClick={() => setColorMode('light')}
              >
                <CIcon className="me-2" icon={cilSun} size="lg" /> Light
              </CDropdownItem>
              <CDropdownItem
                active={colorMode === 'dark'}
                className="d-flex align-items-center"
                as="button"
                type="button"
                onClick={() => setColorMode('dark')}
              >
                <CIcon className="me-2" icon={cilMoon} size="lg" /> Dark
              </CDropdownItem>
              <CDropdownItem
                active={colorMode === 'auto'}
                className="d-flex align-items-center"
                as="button"
                type="button"
                onClick={() => setColorMode('auto')}
              >
                <CIcon className="me-2" icon={cilContrast} size="lg" /> Auto
              </CDropdownItem>
            </CDropdownMenu>
          </CDropdown> */}
        {/* 删除头像
          <li className="nav-item py-1">
            <div className="vr h-100 mx-2 text-body text-opacity-75"></div>
          </li>
          <AppHeaderDropdown /> */}
        {/* </CHeaderNav> */}
      </CContainer>
      {/* 删除main按钮
      <CContainer className="px-4" fluid>
        <AppBreadcrumb />
      </CContainer> */}
    </CHeader>
  )
}

export default AppHeader
