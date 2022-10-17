import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./ShoppingLandingPageFinal.module.css";

const ShoppingLandingPageFinal = () => {
  const navigate = useNavigate();

  const onFashionLogoButtonClick = useCallback(() => {
    navigate("/");
  }, [navigate]);

  const onSignUpButtonClick = useCallback(() => {
    // Please sync "Page2" to the project
  }, []);

  return (
    <div className={styles.shoppingLandingPageFinalDiv}>
      <div className={styles.navigationBarDiv}>
        <div className={styles.navigationInnerBar}>
          <button
            className={styles.fashionLogoButton}
            onClick={onFashionLogoButtonClick}
          >
            <img className={styles.vectorIcon} alt="" src="../vector.svg" />
            <button className={styles.fASHIONButton}>FASHION</button>
          </button>
          <div className={styles.menuRightDiv}>
            <div className={styles.linksContainerDiv}>
              <div className={styles.linksDiv}>
                <button className={styles.cATALOGUEButton}>CATALOGUE</button>
                <button className={styles.fASHIONButton1}>FASHION</button>
                <button className={styles.fAVOURITEButton}>FAVOURITE</button>
                <button className={styles.lIFESTYLEButton}>LIFESTYLE</button>
              </div>
              <button className={styles.component2Button}>
                <div className={styles.rectangleDiv} />
                <div className={styles.lineDiv} />
                <div className={styles.lineDiv1} />
                <div className={styles.lineDiv2} />
              </button>
            </div>
            <div className={styles.frameDiv}>
              <button
                className={styles.signUpButton}
                onClick={onSignUpButtonClick}
              >
                <div className={styles.sIGNUPDiv}>SIGN UP</div>
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.heroDiv}>
        <div className={styles.heroSectionDiv}>
          <div className={styles.heroCTADiv}>
            <div className={styles.rectangleDiv1} />
            <div className={styles.rectangleDiv2} />
            <div className={styles.lETSEXPLOREUNIQUECLOTHES}>
              <p className={styles.lETSP}>LET’S</p>
              <p className={styles.lETSP}>EXPLORE</p>
              <p className={styles.lETSP}>UNIQUE</p>
              <p className={styles.cLOTHESP}>CLOTHES.</p>
            </div>
            <div className={styles.liveForInfluentialAndInnov}>
              Live for Influential and Innovative fashion!
            </div>
            <div className={styles.buttonRowDiv}>
              <img
                className={styles.vectorIcon1}
                alt=""
                src="../vector-9.svg"
              />
              <div className={styles.frameDiv}>
                <div className={styles.frameDiv}>
                  <button className={styles.button}>
                    <div className={styles.shopNowDiv}>Shop Now</div>
                  </button>
                </div>
              </div>
            </div>
          </div>
          <img
            className={styles.womanImageIcon}
            alt=""
            src="../image-1@2x.png"
          />
        </div>
      </div>
      <img className={styles.brandsIcon} alt="" src="../brands@2x.png" />
      <div className={styles.newArrivalsSectionDiv}>
        <div className={styles.newArrivalsDiv}>
          <div className={styles.titleDiv}>
            <img className={styles.leafIcon} alt="" src="../leaf.svg" />
            <div className={styles.titleDiv1}>NEW ARRIVALS</div>
          </div>
          <div className={styles.fashionCardsDiv}>
            <div className={styles.fashionCardDiv}>
              <img
                className={styles.fashionImageIcon}
                alt=""
                src="../fashionimage@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.nameDiv}>{`Hoodies & Sweetshirt`}</div>
                  <div className={styles.exploreNowDiv}>Explore Now!</div>
                </div>
                <img className={styles.arrowIcon} alt="" src="../arrow-1.svg" />
              </div>
            </div>
            <div className={styles.fashionCardDiv1}>
              <img
                className={styles.fashionImageIcon}
                alt=""
                src="../fashionimage1@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.nameDiv1}>{`Coats & Parkas`}</div>
                  <div className={styles.exploreNowDiv1}>Explore Now!</div>
                </div>
                <img className={styles.arrowIcon} alt="" src="../arrow-2.svg" />
              </div>
            </div>
            <div className={styles.fashionCardDiv2}>
              <img
                className={styles.fashionImageIcon}
                alt=""
                src="../fashionimage2@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.nameDiv2}>{`Tees & T-Shirt`}</div>
                  <div className={styles.exploreNowDiv2}>Explore Now!</div>
                </div>
                <img className={styles.arrowIcon} alt="" src="../arrow-3.svg" />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.favouritesDiv}>
        <div className={styles.youngsFavouriteDiv}>
          <div className={styles.titleDiv2}>
            <img className={styles.vectorIcon2} alt="" src="../vector-91.svg" />
            <div className={styles.youngsFavouriteDiv1}>Young’s Favourite</div>
          </div>
          <div className={styles.panelContainerDiv}>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-50@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.trendingOnInstagram}>
                    Trending on instagram
                  </div>
                  <div className={styles.exploreNowDiv3}>Explore Now!</div>
                </div>
                <img className={styles.arrowIcon} alt="" src="../arrow-4.svg" />
              </div>
            </div>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-49@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.allUnder40}>All Under $40</div>
                  <div className={styles.exploreNowDiv4}>Explore Now!</div>
                </div>
                <img className={styles.arrowIcon} alt="" src="../arrow-5.svg" />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.favouritesDiv1}>
        <div className={styles.youngsFavouriteDiv}>
          <div className={styles.titleDiv3}>
            <img className={styles.vectorIcon2} alt="" src="../vector-92.svg" />
            <div className={styles.youngsFavouriteDiv1}>Young’s Favourite</div>
          </div>
          <div className={styles.panelContainerDiv1}>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-501@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.trendingOnInstagram}>
                    Trending on instagram
                  </div>
                  <div className={styles.exploreNowDiv3}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-41.svg"
                />
              </div>
            </div>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-491@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.allUnder40}>All Under $40</div>
                  <div className={styles.exploreNowDiv4}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-51.svg"
                />
              </div>
            </div>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-492@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.allUnder40}>All Under $40</div>
                  <div className={styles.exploreNowDiv4}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-52.svg"
                />
              </div>
            </div>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-493@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.allUnder40}>All Under $40</div>
                  <div className={styles.exploreNowDiv4}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-53.svg"
                />
              </div>
            </div>
          </div>
          <div className={styles.panelContainerDiv1}>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-502@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.trendingOnInstagram}>
                    Trending on instagram
                  </div>
                  <div className={styles.exploreNowDiv3}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-42.svg"
                />
              </div>
            </div>
            <div className={styles.favCardDiv}>
              <img
                className={styles.rectangleIcon}
                alt=""
                src="../rectangle-494@2x.png"
              />
              <div className={styles.navigationInnerBar}>
                <div className={styles.textDiv}>
                  <div className={styles.allUnder40}>All Under $40</div>
                  <div className={styles.exploreNowDiv4}>Explore Now!</div>
                </div>
                <img
                  className={styles.arrowIcon}
                  alt=""
                  src="../arrow-54.svg"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.appDownloadDiv}>
        <div className={styles.vouchersDiv}>
          <div className={styles.downloadAppDiv}>
            <div className={styles.frameDiv3}>
              <div className={styles.dOWNLOADAPPGETTHEVOUCHER}>
                <p className={styles.lETSP}>{`DOWNLOAD APP &`}</p>
                <p className={styles.cLOTHESP}>GET THE VOUCHER!</p>
              </div>
              <div className={styles.get30OffForFirstTransact}>
                <p className={styles.lETSP}>
                  Get 30% off for first transaction using
                </p>
                <p className={styles.cLOTHESP}>
                  Rondovision mobile app for now.
                </p>
              </div>
            </div>
            <div className={styles.frameDiv4}>
              <div className={styles.frameDiv}>
                <button className={styles.frameButton}>
                  <img
                    className={styles.rectangleIcon8}
                    alt=""
                    src="../rectangle-55@2x.png"
                  />
                </button>
              </div>
              <div className={styles.frameDiv}>
                <button className={styles.frameButton}>
                  <img
                    className={styles.rectangleIcon8}
                    alt=""
                    src="../rectangle-18@2x.png"
                  />
                </button>
              </div>
            </div>
          </div>
        </div>
        <img className={styles.image2Icon} alt="" src="../image-2@2x.png" />
      </div>
      <div className={styles.frameDiv7}>
        <div className={styles.communityDiv}>
          <div className={styles.frameDiv8}>
            <form className={styles.formVariant1}>
              <div className={styles.frameDiv9}>
                <div className={styles.frameDiv9}>
                  <form className={styles.frameForm}>
                    <div className={styles.inputChildrenSMALL}>
                      <div className={styles.baseFormContent}>
                        <div className={styles.frameDiv11}>
                          <img
                            className={styles.dateTodayLightIcon}
                            alt=""
                            src="../date-today-light.svg"
                          />
                          <div className={styles.inputDiv}>Input</div>
                        </div>
                        <div className={styles.messageDiv}>Message</div>
                        <div className={styles.div}>0/240</div>
                        <div className={styles.labelDiv}>Name *</div>
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordcacher.svg"
                        />
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordvoir.svg"
                        />
                        <img
                          className={styles.arrowDownIcon}
                          alt=""
                          src="../arrow-down.svg"
                        />
                      </div>
                    </div>
                    <div className={styles.inputChildrenSMALL1}>
                      <div className={styles.baseFormContent}>
                        <div className={styles.frameDiv11}>
                          <img
                            className={styles.dateTodayLightIcon}
                            alt=""
                            src="../date-today-light.svg"
                          />
                          <div className={styles.inputDiv}>Input</div>
                        </div>
                        <div className={styles.messageDiv}>Message</div>
                        <div className={styles.div}>0/240</div>
                        <div className={styles.labelDiv}>Last name *</div>
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordcacher.svg"
                        />
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordvoir.svg"
                        />
                        <img
                          className={styles.arrowDownIcon}
                          alt=""
                          src="../arrow-down.svg"
                        />
                      </div>
                    </div>
                    <div className={styles.inputChildrenSMALL2}>
                      <div className={styles.baseFormContent}>
                        <div className={styles.frameDiv11}>
                          <img
                            className={styles.dateTodayLightIcon}
                            alt=""
                            src="../date-today-light.svg"
                          />
                          <div className={styles.inputDiv}>
                            Customer email filled
                          </div>
                        </div>
                        <div className={styles.messageDiv}>Message</div>
                        <div className={styles.div}>0/240</div>
                        <div className={styles.labelDiv}>Email *</div>
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordcacher2.svg"
                        />
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordvoir2.svg"
                        />
                        <img
                          className={styles.arrowDownIcon}
                          alt=""
                          src="../arrow-down.svg"
                        />
                      </div>
                    </div>
                    <div className={styles.inputChildrenSMALL3}>
                      <div className={styles.baseFormContent3}>
                        <div className={styles.frameDiv14}>
                          <img
                            className={styles.dateTodayLightIcon}
                            alt=""
                            src="../date-today-light.svg"
                          />
                          <div className={styles.inputDiv3}>Input</div>
                        </div>
                        <div className={styles.messageDiv}>Message</div>
                        <div className={styles.div}>0/240</div>
                        <div className={styles.labelDiv}>Object *</div>
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordcacher2.svg"
                        />
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordvoir2.svg"
                        />
                        <img
                          className={styles.arrowDownIcon3}
                          alt=""
                          src="../arrow-down.svg"
                        />
                      </div>
                    </div>
                    <div className={styles.inputChildrenSMALL4}>
                      <div className={styles.baseFormContent4}>
                        <div className={styles.frameDiv14}>
                          <img
                            className={styles.dateTodayLightIcon}
                            alt=""
                            src="../date-today-light.svg"
                          />
                          <div className={styles.inputDiv}>Text area</div>
                        </div>
                        <div className={styles.messageDiv4}>Message</div>
                        <div className={styles.div4}>0/240</div>
                        <div className={styles.labelDiv}>Message *</div>
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordcacher2.svg"
                        />
                        <img
                          className={styles.iconsPasswordCacher}
                          alt=""
                          src="../iconspasswordvoir2.svg"
                        />
                        <img
                          className={styles.arrowDownIcon}
                          alt=""
                          src="../arrow-down.svg"
                        />
                      </div>
                    </div>
                    <div className={styles.frameDiv16}>
                      <div className={styles.frameDiv17}>
                        <img
                          className={styles.uilcheckIcon}
                          alt=""
                          src="../uilcheck.svg"
                        />
                        <div
                          className={styles.iReadTheTermsConditions}
                        >{`I read the terms & conditions and the Privacy policy`}</div>
                      </div>
                    </div>
                  </form>
                </div>
              </div>
              <button className={styles.button1}>
                <div className={styles.sENDDiv}>SEND</div>
              </button>
            </form>
          </div>
        </div>
      </div>
      <div className={styles.frameDiv18}>
        <div className={styles.communityDiv1}>
          <div className={styles.frameDiv19}>
            <div className={styles.miniSizeDiv}>
              <div className={styles.miniCard1}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Glen Williams</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    7 Skills of Highly Effective Programmers
                  </div>
                  <div className={styles.frameDiv20}>
                    <div className={styles.ourTeamWasInspiredByTheS}>
                      Our team was inspired by the seven skills of highly
                      effective programmers created by the TechLead. We wanted
                      to provide our own take on the topic...
                    </div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>3 days ago</div>
                <div className={styles.pRODUCTIVITYDiv}>PRODUCTIVITY</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px--laptop.svg"
                />
              </div>
              <div className={styles.miniCard2}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Patricia Kemp</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar1@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    SMM starter pack, part 2: content promotion
                  </div>
                  <div className={styles.frameDiv20}>
                    <div
                      className={styles.ourTeamWasInspiredByTheS}
                    >{`This is the second part of the SMM starter pack series of articles. If you made it this far, you must be willing to learn about promoting business through social media. `}</div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>17 days ago</div>
                <div className={styles.pRODUCTIVITYDiv}>MEDIA</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv1} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px--social-media.svg"
                />
              </div>
              <div className={styles.miniCard1}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Lisa Barnes</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar2@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    11 Things I Wish I Knew When I Started My Business
                  </div>
                  <div className={styles.frameDiv20}>
                    <div className={styles.ourTeamWasInspiredByTheS}>
                      Here are 11 things I wish I knew when I started my
                      business. I hope they will save you some time and some
                      anguish because (experience is a good teacher here)...
                    </div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>1 month ago</div>
                <div className={styles.pRODUCTIVITYDiv}>BUSINESS</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv2} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px---business.svg"
                />
              </div>
            </div>
            <div className={styles.miniSizeDiv1}>
              <div className={styles.miniCard1}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Glen Williams</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar3@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    7 Skills of Highly Effective Programmers
                  </div>
                  <div className={styles.frameDiv20}>
                    <div className={styles.ourTeamWasInspiredByTheS}>
                      Our team was inspired by the seven skills of highly
                      effective programmers created by the TechLead. We wanted
                      to provide our own take on the topic...
                    </div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>3 days ago</div>
                <div className={styles.pRODUCTIVITYDiv}>PRODUCTIVITY</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px--laptop1.svg"
                />
              </div>
              <div className={styles.miniCard2}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Patricia Kemp</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar4@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    SMM starter pack, part 2: content promotion
                  </div>
                  <div className={styles.frameDiv20}>
                    <div
                      className={styles.ourTeamWasInspiredByTheS}
                    >{`This is the second part of the SMM starter pack series of articles. If you made it this far, you must be willing to learn about promoting business through social media. `}</div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>17 days ago</div>
                <div className={styles.pRODUCTIVITYDiv}>MEDIA</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv1} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px--social-media1.svg"
                />
              </div>
              <div className={styles.miniCard1}>
                <div className={styles.cardBackgroundDiv} />
                <div className={styles.readMore}>
                  <span className={styles.readMore1}>{`Read more `}</span>
                  <span>🡢</span>
                </div>
                <div className={styles.glenWilliamsDiv}>Lisa Barnes</div>
                <img
                  className={styles.avatarIcon}
                  alt=""
                  src="../avatar5@2x.png"
                />
                <div className={styles.headingDescription}>
                  <div className={styles.skillsOfHighlyEffectivePro}>
                    11 Things I Wish I Knew When I Started My Business
                  </div>
                  <div className={styles.frameDiv20}>
                    <div className={styles.ourTeamWasInspiredByTheS}>
                      Here are 11 things I wish I knew when I started my
                      business. I hope they will save you some time and some
                      anguish because (experience is a good teacher here)...
                    </div>
                  </div>
                </div>
                <div className={styles.daysAgoDiv}>1 month ago</div>
                <div className={styles.pRODUCTIVITYDiv}>BUSINESS</div>
                <div className={styles.frameDiv21}>
                  <div className={styles.iconBackgroundDiv2} />
                </div>
                <img
                  className={styles.icon120pxLaptop}
                  alt=""
                  src="../icon-120px---business1.svg"
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.communityDiv2}>
        <div className={styles.communityInnerDiv}>
          <div className={styles.joinDiv}>
            <div className={styles.jOINSHOPPINGCOMMUNITYTOGET}>
              JOIN SHOPPING COMMUNITY TO GET MONTHLY PROMO
            </div>
            <div className={styles.typeYourEmailDownBelowAnd}>
              Type your email down below and be young wild generation
            </div>
          </div>
          <div className={styles.formDiv}>
            <input
              className={styles.addYourEmailHere}
              type="text"
              placeholder="Add your email here"
            />
            <div className={styles.frameDiv}>
              <button className={styles.button2}>
                <div className={styles.sENDDiv}>SEND</div>
              </button>
            </div>
          </div>
        </div>
      </div>
      <div className={styles.footerDiv}>
        <div className={styles.innerFooterDiv}>
          <div className={styles.fashionDiv}>
            <div className={styles.fASHIONDiv}>FASHION</div>
            <div className={styles.completeYourStyleWithAweso}>
              Complete your style with awesome clothes from us.
            </div>
            <div className={styles.logosDiv}>
              <div className={styles.groupDiv}>
                <div className={styles.rectangleDiv3} />
                <img
                  className={styles.vectorIcon4}
                  alt=""
                  src="../vector1.svg"
                />
              </div>
              <div className={styles.groupDiv}>
                <div className={styles.rectangleDiv3} />
                <img
                  className={styles.vectorIcon5}
                  alt=""
                  src="../vector2.svg"
                />
              </div>
              <div className={styles.groupDiv}>
                <div className={styles.rectangleDiv3} />
                <img
                  className={styles.vectorIcon6}
                  alt=""
                  src="../vector3.svg"
                />
              </div>
              <div className={styles.groupDiv}>
                <div className={styles.rectangleDiv3} />
                <b className={styles.inB}>in</b>
              </div>
            </div>
          </div>
          <div className={styles.linksDiv1}>
            <div className={styles.column1Div}>
              <div className={styles.companyDiv}>Company</div>
              <div className={styles.aboutDiv}>About</div>
              <div className={styles.contactUsDiv}>Contact us</div>
              <div className={styles.supportDiv}>Support</div>
              <div className={styles.careersDiv}>Careers</div>
            </div>
            <div className={styles.column2Div}>
              <div className={styles.quickLinkDiv}>Quick Link</div>
              <div className={styles.shareLocationDiv}>Share Location</div>
              <div className={styles.ordersTrackingDiv}>Orders Tracking</div>
              <div className={styles.sizeGuideDiv}>Size Guide</div>
              <div className={styles.fAQsDiv}>FAQs</div>
            </div>
            <div className={styles.column3Div}>
              <div className={styles.legalDiv}>Legal</div>
              <div
                className={styles.termsConditions}
              >{`Terms & conditions`}</div>
              <div className={styles.privacyPolicyDiv}>Privacy Policy</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ShoppingLandingPageFinal;
